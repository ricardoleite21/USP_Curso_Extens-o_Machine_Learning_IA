from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')

# classes de interesse: pessoa (0) e carro (2)
TARGET_CLASSES = [0, 2]

cap = cv2.VideoCapture('vru.mp4')

# configura o VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('saida_vru_detectado.mp4', fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for box in results.boxes:
        cls = int(box.cls)
        if cls in TARGET_CLASSES:
            label = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("YOLOv8 - Pessoa e Carro", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
