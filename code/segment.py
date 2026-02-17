from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np

model = YOLO("yolo26n-seg.pt")
cap = cv2.VideoCapture(0)

# Setup Matplotlib for a separate "analysis" window
plt.ion() # Turn on interactive mode
fig, ax = plt.subplots()

while True:
    ret, frame = cap.read()
    if not ret: break

    # Inference
    results = model(frame, stream=False) # stream=False for easier single-frame access

    for r in results:
        # 1. Print Shapes to Terminal
        if r.masks is not None:
            # Mask data is usually [number_of_objects, height, width]
            print(f"Detected {len(r.masks)} objects.")
            print(f"Masks shape: {r.masks.data.shape}") 
            print(f"Bounding Boxes shape: {r.boxes.xyxy.shape}")
            print("-" * 20)

            # 2. Show on another "terminal" (Matplotlib window)
            # We'll take the first mask detected and show it
            mask_data = r.masks.data[0].cpu().numpy() # Move to CPU and convert to numpy
            ax.clear()
            ax.imshow(mask_data, cmap='gray')
            ax.set_title("Primary Object Mask Tensor")
            plt.pause(0.01) # Small pause to allow Matplotlib to update

        # 3. Live OpenCV Pop-up
        cv2.imshow("OpenCV Live Feed", r.plot())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()