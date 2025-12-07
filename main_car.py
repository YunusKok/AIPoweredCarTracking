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
from datetime import datetime

class ThreadedCamera:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Buffer'ı 1 yap
        self.success, self.frame = self.capture.read()
        self.stop_thread = False
        # Arka planda okuma işlemini başlat
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while True:
            if self.stop_thread: break
            # Sürekli en son kareyi oku
            self.success, self.frame = self.capture.read()
            
    def read(self):
        # Ana döngü istediğinde son kareyi ver
        return self.success, self.frame

    def release(self):
        self.stop_thread = True
        self.capture.release()
# initialize firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

def send_data_to_firebase(class_name, track_id):
    doc_ref = db.collection(u'detected_vehicles').document()
    doc_ref.set({
        u'class': class_name,
        u'track_id': track_id,
        u'timestamp': datetime.utcnow(), 
        u'location': u'CAM1' # if there are multiple cameras, specify location
    })
    print(f"Data sent to Firebase for {class_name} with ID {track_id}") # confirmation message at terminal

# assist function definition
def get_line_side(x, y, line_start, line_end): # use to determine which side of the line the object is on
    return np.sign((line_end[0] - line_start[0])*(y - line_start[1]) - 
                   (line_end[1] - line_start[1])*(x - line_start[0]))

# define model
model = YOLO("yolov8n.pt")

url = "http://192.168.1.111:4747/video" # path to ip cam

# video capture
cap = ThreadedCamera(url) # video path for testing it wiill be changed later with camera (0 for default camera)

success, frame = cap.read()
if not success:
    exit("Video could not be opened")

#frame = cv2.resize(frame, (0,0), fx = 0.6, fy = 0.6)
frame_height, frame_width = frame.shape[:2]

# define crossing line
line_start = (int(frame_height*0.5), frame_height)
line_end =  (frame_width, int(frame_width*0.2))

# object types / counters
counts = {"car":0, "truck":0, "bus":0, "motorcycle": 0, "bicycle": 0}
counted_ids = set()
object_last_side = {}

# vehicle counting loop using YOLO
while True: 

    success, frame = cap.read()
    if not success:
        exit("Video could not be opened")

    #frame = cv2.resize(frame, (0,0), fx = 0.6, fy = 0.6) # resize frame
    
    # get frame dimensions
    frame_height, frame_width = frame.shape[:2]
    line_start = (0, int(frame_height*0.75))
    line_end =  (frame_width, int(frame_height*0.75))
    
    # tracking (object tracking)
    results = model.track(frame, persist=True, stream=False, conf = 0.25, iou = 0.45, tracker = "bytetrack.yaml", verbose=False) # using ByteTrack for tracking

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
                    try:
                        send_data_to_firebase(class_name, track_id)
                    except Exception as e:
                        print(f"Error sending data to Firebase. {e}") 

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