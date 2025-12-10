"""
Problem definition: vehicle counting from video using YOLO
    - car
    - truck
    - bus

After tracking with YOLO, count vehicles that cross a defined line.

data: https://www.kaggle.com/datasets/benjaminguerrieri/car-detection-videos?select=IMG_5268.MOV

"""
# import libraries
import threading
import time
import cv2 # opencv
import numpy as np
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timezone
import queue # Import queue for thread-safe communication


IP_CAM_URL = "http://192.168.137.218:4747/video"

# --- Performance Fix: Firebase Worker Setup ---
# Create a queue to hold data to be sent
firebase_queue = queue.Queue()

# Initialize firebase
try:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("✅ Firebase connected successfully.")
except Exception as e:
    print(f"❌ Firebase connection failed: {e}")

def firebase_worker():
    """
    Worker thread function that constantly checks the queue 
    and sends data to Firebase asynchronously.
    """
    print("--- FIREBASE WORKER STARTED ---")
    while True:
        # Get data from the queue (blocks until data is available)
        data = firebase_queue.get()
        
        if data is None: # Sentinel to stop the thread
            break
            
        try:
            class_name = data['class_name']
            track_id = data['track_id']
            
            doc_ref = db.collection(u'detected_vehicles').document()
            doc_ref.set({
                u'class': class_name,
                u'track_id': track_id,
                u'timestamp': datetime.now(timezone.utc), 
                u'location': u'CAM1' 
            })
            print(f"🚀 Data sent to Firebase: {class_name} (ID: {track_id})") 
        except Exception as e:
            print(f"❌ Error sending data to Firebase: {e}")
        finally:
            # Mark the task as done
            firebase_queue.task_done()

# Start the Firebase worker thread as a daemon
threading.Thread(target=firebase_worker, daemon=True).start()
# -----------------------------------------------

class ThreadedCamera:
    """
    Reads frames in a separate thread to prevent blocking the main loop
    and reduces lag for IP Cameras. Includes safe shutdown.
    """
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Buffer to 1
        self.success, self.frame = self.capture.read()
        self.stop_thread = False
        
        # Start background frame reading only if capture is successful
        if self.success:
            self.thread = threading.Thread(target=self.update, daemon=True)
            self.thread.start()

    def update(self):
        while True:
            if self.stop_thread: 
                break
            try:
                # Check if capture is open before reading
                if self.capture.isOpened():
                    self.success, self.frame = self.capture.read()
                    if not self.success:
                        self.stop_thread = True # Stop if stream fails
                else:
                    self.stop_thread = True
            except Exception as e:
                self.stop_thread = True
                break
            
    def read(self):
        # Return the latest frame when requested
        return self.success, self.frame

    def release(self):
        self.stop_thread = True
        # Wait briefly for the thread to exit safely (Prevent Race Condition)
        time.sleep(0.1) 
        if self.capture.isOpened():
            self.capture.release()

# assist function definition
def get_line_side(x, y, line_start, line_end): # use to determine which side of the line the object is on
    return np.sign((line_end[0] - line_start[0])*(y - line_start[1]) - 
                   (line_end[1] - line_start[1])*(x - line_start[0]))

# define model
model = YOLO("yolov8n.pt")

print(f"Connecting to IP Camera: {IP_CAM_URL}")
# video capture
cap = ThreadedCamera(IP_CAM_URL) # IP camera

# Check initial connection
if not cap.success:
    print("❌ ERROR: Could not connect to IP Camera.")
    print("   - Check if the phone and computer are on the same Wi-Fi.")
    print("   - Check the IP address.")
    exit()

success, frame = cap.read()
frame_height, frame_width = frame.shape[:2]

# define crossing line (Initial definition)
# Fixed syntax error and logic here
line_start = (int(frame_width * 0.5), int(frame_height * 0.5))
line_end = (frame_width, frame_height)

# object types / counters
counts = {"car":0, "truck":0, "bus":0, "motorcycle": 0, "bicycle": 0}
counted_ids = set()
object_last_side = {}

print("System started. Processing live stream...")

# vehicle counting loop using YOLO
while True: 

    success, frame = cap.read()
    if not success:
        print("Failed to read frame or stream ended.")
        break

    # get frame dimensions
    frame_height, frame_width = frame.shape[:2]
    
    # Redefining line based on frame size inside loop (ensure consistency)
    # Line: Center of screen -> Bottom Right Corner
    line_start = (int(frame_width * 0.5), int(frame_height * 0.5))
    line_end = (frame_width, frame_height)
    
    # tracking (object tracking)
    # Added verbose=False to reduce console spam
    results = model.track(frame, persist=True, stream=False, conf=0.25, iou=0.45, tracker="bytetrack.yaml", verbose=False) 

    if results[0].boxes.id is not None: # if there are tracked objects
        ids = results[0].boxes.id.int().tolist() # get all ids
        classes = results[0].boxes.cls.int().tolist() # get all classes
        xyxy = results[0].boxes.xyxy # coordinates

        for i, box in enumerate(xyxy):
            cls_id = classes[i]
            track_id = ids[i]
            class_name = model.names[cls_id]
            
            if class_name not in counts:
                continue
                
            x1, y1, x2, y2 = map(int, box) # get the box x and y coordinates
            # find center
            cx = int((x1+x2)/2)
            cy = int((y1+y2)/2)

            current_side = get_line_side(cx, cy, line_start, line_end) # which side of the line the vehicle is currently on
            previous_side = object_last_side.get(track_id, None) # where it was in the previous frame
            object_last_side[track_id] = current_side # update last side

            if previous_side is not None and previous_side != current_side: # crossing detection
                if track_id not in counted_ids:
                    counted_ids.add(track_id)
                    counts[class_name] += 1 # increment count by 1
                    
                    # --- Performance Fix: Enqueue Data ---
                    # Instead of sending directly, put into queue. This is non-blocking.
                    payload = {'class_name': class_name, 'track_id': track_id}
                    firebase_queue.put(payload)
                    print(f"Queued -> {class_name} (ID: {track_id})")
                    # -------------------------------------

            # draw bounding boxes
            cv2.rectangle(frame, (x1, y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"{class_name} ID: {track_id}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

    # draw crossing line
    cv2.line(frame, line_start, line_end, (0, 0, 255), 2)

    # display counters
    y_offset = 30
    for cls, count in counts.items():
        text = f"{cls}: {count}"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30

    # show frame
    cv2.imshow("Vehicle tracking and counting", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Check for remaining data before exiting
if not firebase_queue.empty():
    print("Exiting... Waiting for remaining data to be sent...")
    firebase_queue.join()
    
print("Program finished successfully.")