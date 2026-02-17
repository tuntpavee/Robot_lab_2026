import cv2
import numpy as np
from google import genai
from ultralytics import YOLO
from PIL import Image

# 1. Initialize Gemini Client
# Ensure GEMINI_API_KEY is in your environment variables
client = genai.Client(api_key= "AIzaSyBHBG9X_BQG50MQDwfgSa4X4OUz2pZr1IQ")

# 2. Load YOLO
model = YOLO("yolo26n-seg.pt")
cap = cv2.VideoCapture(0)

# Variables to store the data we want to send to LLM
captured_rgb_crop = None
captured_mask_crop = None
captured_class = "Unknown"

print("--- YIMBOT VISION SYSTEM ---")
print("Press 'q' to capture the current object and analyze its array/shape.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Run YOLO
    results = model(frame, stream=False)
    annotated_frame = frame.copy()

    # ---------------------------------------------------------
    # SAFETY CHECK: Only proceed if detections exist
    # ---------------------------------------------------------
    # This prevents the "IndexError: index 0 is out of bounds"
    if results[0].boxes is not None and len(results[0].boxes) > 0 and results[0].masks is not None:
        
        r = results[0]
        
        # 1. Get the Box Coordinates (for cropping)
        box = r.boxes.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box

        # 2. Get the Class Name
        cls_id = int(r.boxes.cls[0])
        captured_class = model.names[cls_id]

        # 3. Get the Mask Array (The "Array" you wanted)
        # YOLO returns masks often in lower res (e.g., 160x160 or 640x640)
        # We process it to match the original frame
        raw_mask = r.masks.data[0].cpu().numpy()          # The raw float array
        
        # Resize mask to match the frame size using OpenCV
        # This ensures the array maps 1:1 to the image pixels
        mask_resized = cv2.resize(raw_mask, (frame.shape[1], frame.shape[0]))
        
        # Crop the Frame (Color)
        captured_rgb_crop = frame[y1:y2, x1:x2]
        
        # Crop the Mask (The Array Visualized)
        # We crop the mask using the exact same box coordinates
        captured_mask_crop = mask_resized[y1:y2, x1:x2]

        # Visualization for the loop
        annotated_frame = r.plot()

    # Show the live feed
    cv2.imshow("YOLO + Gemini Link", annotated_frame)

    # Break loop on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ---------------------------------------------------------
# SECTION 2: SENDING ARRAY & IMAGE TO GEMINI
# ---------------------------------------------------------
if captured_rgb_crop is not None and captured_mask_crop is not None:
    print(f"\n[SYSTEM] Analyzing object: {captured_class}...")

    # 1. Convert Color Image for Gemini
    rgb_pil = Image.fromarray(cv2.cvtColor(captured_rgb_crop, cv2.COLOR_BGR2RGB))

    # 2. Convert Mask Array for Gemini
    # We convert the 0-1 float array to a 0-255 grayscale image so Gemini can "see" the array structure
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

    # 4. Send to Gemini (Multimodal: Text + Image 1 + Image 2)
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[prompt, rgb_pil, mask_pil]
    )

    print("\n" + "="*40)
    print(f"GEMINI ANALYSIS OF {captured_class.upper()}")
    print("="*40)
    print(response.text)

else:
    print("[SYSTEM] No object detected. Nothing sent to Gemini.")