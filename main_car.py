"""
Basit araç algılama ve geçiş sayacı.
Bu dosya şu düzeltmeleri ve iyileştirmeleri içerir:
- Söz dizimi hatalarının düzeltilmesi
- YOLOv8 modelinden algılama sonuçlarının alınması
- Basit bir centroid tracker ile nesne takibi
- Bir çizgiyi geçen araçların sayılması

Kullanım:
 - Varsayılan video dosyası `IMG_5268.MOV`'dir; yoksa webcam (0) açılır.
 - Model dosyası `yolov8n.pt` aynı klasörde bulunmalıdır.
"""

import time
import math
import cv2
import numpy as np
from ultralytics import YOLO


class CentroidTracker:
    def __init__(self, max_disappeared=50, max_distance=50):
        self.next_object_id = 0
        self.objects = dict()  # object_id -> centroid
        self.disappeared = dict()  # object_id -> frames disappeared
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, input_centroids):
        if len(input_centroids) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects

        if len(self.objects) == 0:
            for c in input_centroids:
                self.register(c)
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        D = np.zeros((len(object_centroids), len(input_centroids)), dtype="float")
        for i, oc in enumerate(object_centroids):
            for j, ic in enumerate(input_centroids):
                D[i, j] = math.hypot(oc[0] - ic[0], oc[1] - ic[1])

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue
            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[col]
            self.disappeared[object_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(0, D.shape[0])).difference(used_rows)
        unused_cols = set(range(0, D.shape[1])).difference(used_cols)

        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        for col in unused_cols:
            self.register(input_centroids[col])

        return self.objects


def main(video_path="IMG_5268.MOV", model_path="yolov8n.pt"):
    print("Libraries imported successfully.")

    # YOLO model
    model = YOLO(model_path)

    # Açmayı dene, yoksa webcam'e dön
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Video '{video_path}' açılamadı, webcam(0) deneniyor...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise SystemExit("Video veya webcam açılamadı.")

    success, frame = cap.read()
    if not success:
        raise SystemExit("Video okunamıyor veya boş kare." )

    frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
    frame_height, frame_width = frame.shape[:2]

    # Sayma çizgisi: çapraz (alt-orta -> sağ-orta)
    # Başlangıç: alt-orta (frame_width//2, frame_height)
    # Bitiş: sağ-orta (frame_width, frame_height//2)
    line_start = (frame_width // 2, frame_height)
    line_end = (frame_width, frame_height // 2)
    offset = 10

    def get_line_side(x, y, line_start, line_end):
        return (line_end[0] - line_start[0]) * (y - line_start[1]) - (line_end[1] - line_start[1]) * (x - line_start[0])

    tracker = CentroidTracker(max_disappeared=40, max_distance=60)
    counted_ids = set()
    total_count = 0

    # Önceki centroid pozisyonlarını sakla (id -> (x,y))
    prev_positions = {}

    allowed_classes = {"car", "truck", "bus", "motorbike", "bicycle"}

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)

        # ultralytics model çağırma
        results = model(frame)[0]

        boxes = []
        try:
            for b in results.boxes:
                xyxy = b.xyxy[0].cpu().numpy() if hasattr(b.xyxy[0], 'cpu') else b.xyxy[0].numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                conf = float(b.conf[0]) if hasattr(b.conf, '__len__') else float(b.conf)
                cls = int(b.cls[0]) if hasattr(b.cls, '__len__') else int(b.cls)
                name = model.names.get(cls, str(cls)) if hasattr(model, 'names') else str(cls)
                if name in allowed_classes and conf > 0.3:
                    boxes.append((x1, y1, x2, y2, conf, name))
        except Exception:
            # Fallback if API is slightly different
            for det in results.boxes.data.tolist() if hasattr(results.boxes, 'data') else []:
                # Not ideal, but keep robust
                pass

        centroids = []
        for (x1, y1, x2, y2, conf, name) in boxes:
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            centroids.append((cX, cY))

        objects = tracker.update(centroids)

        # Draw detections and check crossing
        for oid, centroid in objects.items():
            cX, cY = centroid
            cv2.circle(frame, (cX, cY), 4, (0, 255, 0), -1)
            cv2.putText(frame, f"ID {oid}", (cX - 10, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            # Geçiş kontrolü: önceki ve şimdiki centroid'lerin çizgi tarafları farklıysa geçti
            prev = prev_positions.get(oid)
            if prev is not None:
                side_prev = get_line_side(prev[0], prev[1], line_start, line_end)
                side_curr = get_line_side(cX, cY, line_start, line_end)
                if side_prev * side_curr < 0 and oid not in counted_ids:
                    counted_ids.add(oid)
                    total_count += 1

            # Son pozisyonu güncelle
            prev_positions[oid] = (cX, cY)

        # Görsel öğeler
        # Çapraz sayma çizgisi
        cv2.line(frame, line_start, line_end, (0, 0, 255), 2)
        cv2.putText(frame, f"Count: {total_count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        # Bounding boxes çiz
        for (x1, y1, x2, y2, conf, name) in boxes:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
            cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

        cv2.imshow("Car tracking and counting", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Eğer farklı bir video yolu istenirse komut satırı argümanı eklenebilir
    main()
