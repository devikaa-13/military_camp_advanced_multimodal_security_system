from ultralytics import YOLO
import pandas as pd
model = YOLO("/Users/devika/Downloads/best.pt")


results=model.predict(source='/Users/devika/Downloads/knife.jpg', save=True, save_txt=True, save_conf=True,save_crop=True,vid_stride=15, iou=0.7, show=True)
# Print class IDs
all_outputs=[]
result = results[0]
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = [x for x in box.xyxy[0].tolist()]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        all_outputs.append([
        result.path,x1, y1, x2, y2, result.names[class_id], prob
        ])
df = pd.DataFrame(all_outputs, columns=["image","x1", "y1", "x2", "y2", "class_id", "prob"])
df.to_csv("output.csv", index=False)