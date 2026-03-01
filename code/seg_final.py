import cv2
import numpy as np
from google import genai
from ultralytics import YOLO
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import os

# 1. Initialize Gemini Client
# Ensure your API key is correct
client = genai.Client(api_key="AIzaSyC4CnjfJtxRxUeigDqP_JN7QCheqoS0530")

# 2. Load YOLO
model = YOLO("yolo26n-seg.pt")
cap = cv2.VideoCapture(0)

# Tracking for Confusion Matrix
y_true = []
y_pred = []

# Variables to store the data we want to send to LLM
captured_rgb_crop = None
captured_mask_crop = None
captured_class = "Unknown"

print("--- YIMBOT VISION SYSTEM ---")
print("Controls:")
print("  'c' -> Capture & Validate (Tell the AI if it's right for the matrix)")
print("  'q' -> Quit, analyze the last object with Gemini, and show Report")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Run YOLO
    results = model(frame, stream=False, verbose=False)
    annotated_frame = frame.copy()

    # SAFETY CHECK: Only proceed if detections and masks exist
    if results[0].boxes is not None and len(results[0].boxes) > 0 and results[0].masks is not None:
        
        r = results[0]
        
        # 1. Get the Box Coordinates (for cropping)
        box = r.boxes.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box

        # 2. Get the Class Name (Always track the most confident one)
        cls_id = int(r.boxes.cls[0])
        captured_class = model.names[cls_id]

        # 3. Get the Mask Array and Resize
        raw_mask = r.masks.data[0].cpu().numpy()          
        mask_resized = cv2.resize(raw_mask, (frame.shape[1], frame.shape[0]))
        
        # 4. Update the crops constantly so we have the latest one when 'q' is pressed
        # Safety check to prevent empty crops if bounding box is out of frame bounds
        if y2 > y1 and x2 > x1:
            captured_rgb_crop = frame[y1:y2, x1:x2]
            captured_mask_crop = mask_resized[y1:y2, x1:x2]

        # Visualization for the loop
        annotated_frame = r.plot()
        cv2.putText(annotated_frame, f"YOLO: {captured_class} | 'c' to Validate, 'q' to Quit", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the live feed
    cv2.imshow("YOLO + Gemini Link", annotated_frame)
    key = cv2.waitKey(1) & 0xFF

    # --- VALIDATION STEP ---
    if key == ord('c') and len(results[0].boxes) > 0:
        print(f"\n[VALIDATION] YOLO thinks this is: {captured_class}")
        actual = input(f"What is this actually? (Press Enter if {captured_class} is correct): ").strip()
        
        ground_truth = actual if actual != "" else captured_class
        y_true.append(ground_truth)
        y_pred.append(captured_class)
        print(f"Recorded: True={ground_truth}, Pred={captured_class}")

    # Break loop on 'q'
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ---------------------------------------------------------
# SECTION 2: SENDING ARRAY & IMAGE TO GEMINI
# ---------------------------------------------------------
if captured_rgb_crop is not None and captured_mask_crop is not None and captured_rgb_crop.size > 0:
    print(f"\n[SYSTEM] Analyzing last captured object: {captured_class} with Gemini...")

    # 1. Convert Color Image for Gemini
    rgb_pil = Image.fromarray(cv2.cvtColor(captured_rgb_crop, cv2.COLOR_BGR2RGB))

    # 2. Convert Mask Array for Gemini
    mask_uint8 = (captured_mask_crop * 255).astype('uint8')
    mask_pil = Image.fromarray(mask_uint8)

    # 3. Construct the Prompt with Class Context
    prompt = (
        f"I am analyzing a robotic perception task. "
        f"The object class is identified as '{captured_class}'.\n\n"
        "I have provided two images:\n"
        "1. The visual appearance (Color Crop).\n"
        "2. The segmentation mask array visualized (Black & White).\n\n"
        "Based on the exact shape of the mask array and the visual texture:\n"
        "- Describe the object's condition.\n"
        "- Verify if the segmentation mask accurately covers the object or if it missed parts."
    )

    # 4. Send to Gemini
    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[prompt, rgb_pil, mask_pil]
        )
        print("\n" + "="*40)
        print(f"GEMINI ANALYSIS OF {captured_class.upper()}")
        print("="*40)
        print(response.text)
    except Exception as e:
        print(f"\n[ERROR] Gemini API failed: {e}")

else:
    print("\n[SYSTEM] No valid object captured for Gemini analysis.")


# ---------------------------------------------------------
# SECTION 3: GENERATE CONFUSION MATRIX
# ---------------------------------------------------------
if len(y_true) > 0:
    print("\n" + "="*40)
    print("LIVE SESSION PERFORMANCE REPORT")
    print("="*40)
    
    labels = sorted(list(set(y_true + y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"True:{l}" for l in labels], columns=[f"Pred:{l}" for l in labels])
    
    print("\nConfusion Matrix:")
    print(cm_df)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=labels))
else:
    print("\n[SYSTEM] No validation samples collected for the Confusion Matrix.")