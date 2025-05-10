import cv2
from deepface import DeepFace
import os
import csv
from datetime import datetime

print("📷 Starting webcam with attendance logger. Press 'q' to quit.")

cap = cv2.VideoCapture(0)
attendance_log = set()

# Create attendance CSV
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"attendance_{timestamp}.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Name", "Time"])

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    try:
        results = DeepFace.find(img_path=frame, db_path="test_faces/known", model_name="ArcFace", enforce_detection=False, silent=True)

        df = results[0] if isinstance(results, list) and len(results) > 0 else None
        if df is not None and not df.empty:
            top_match = df.iloc[0]
            label = os.path.basename(os.path.dirname(top_match["identity"]))
            confidence = 1 - top_match["distance"]

            if confidence > 0.3:
                if label not in attendance_log:
                    attendance_log.add(label)
                    with open(csv_filename, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([label, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                    print(f"📝 Marked present: {label}")

                text = f"{label} ({confidence:.2f})"
                color = (0, 255, 0)
            else:
                text = "Unknown"
                color = (0, 0, 255)
        else:
            text = "No match"
            color = (0, 0, 255)

        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    except Exception as e:
        cv2.putText(frame, "Detection error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Attendance Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"✅ Attendance saved to: {csv_filename}")