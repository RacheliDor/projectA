import os
import math
import cv2
from ultralytics import YOLO

# --- הגדרות ---
# ודאי שהנתיבים האלו מדוייקים!
LOG_FILE_PATH = '/home/user_104/RR/drone_full_scan_results/positions_log.txt'
IMAGE_FOLDER = '/home/user_104/RR/drone_full_scan_results/3' 
MODEL_PATH = '/home/user_104/RR/runs/drone_model_final_small/weights/best.pt'
OUTPUT_FILE = '/home/user_104/RR/final_human_locations.csv'

# נתוני מצלמה
FOV = 90.0
DRONE_ALTITUDE = 20.0 

def parse_log_file(log_path):
    coords_db = {}
    if not os.path.exists(log_path):
        print(f"❌ ERROR: Log file NOT found at: {log_path}")
        return {}

    print(f"✅ Found log file at: {log_path}")
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
            # דילוג על כותרות
            for line in lines[2:]:
                parts = line.split('|')
                if len(parts) >= 6:
                    x = float(parts[2].strip())
                    y = float(parts[3].strip())
                    filename = parts[5].strip() 
                    coords_db[filename] = {'x': x, 'y': y}
        print(f"✅ Successfully loaded {len(coords_db)} positions from log.")
    except Exception as e:
        print(f"❌ ERROR parsing log file: {e}")
        
    return coords_db

def pixel_to_meters(x_center, y_center, img_w, img_h, altitude):
    fov_rad = math.radians(FOV)
    ground_width_meters = 2 * altitude * math.tan(fov_rad / 2)
    ground_height_meters = ground_width_meters * (img_h / img_w)
    
    x_dist_px = x_center - (img_w / 2)
    y_dist_px = y_center - (img_h / 2)
    
    meters_per_pixel_x = ground_width_meters / img_w
    meters_per_pixel_y = ground_height_meters / img_h
    
    # --- התיקון: השורות שהיו חסרות ---
    offset_x_meters = x_dist_px * meters_per_pixel_x
    offset_y_meters = y_dist_px * meters_per_pixel_y
    # --------------------------------
    
    real_offset_forward = -offset_y_meters 
    real_offset_right = offset_x_meters    
    
    return real_offset_forward, real_offset_right

def main():
    print("\n--- DIAGNOSTICS START ---")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Model file not found at {MODEL_PATH}")
        return

    if not os.path.exists(IMAGE_FOLDER):
        print(f"❌ ERROR: Image folder not found at {IMAGE_FOLDER}")
        return
        
    positions = parse_log_file(LOG_FILE_PATH)
    if not positions: 
        print("❌ STOPPING: Could not load positions.")
        return

    print("--- LOADING MODEL ---")
    model = YOLO(MODEL_PATH)
    
    print(f"--- SAVING TO: {OUTPUT_FILE} ---")
    
    found_count = 0
    with open(OUTPUT_FILE, 'w') as f_out:
        f_out.write("Image_File,Drone_X,Drone_Y,Human_Real_X,Human_Real_Y\n")
        
        image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith('.png')]
        image_files.sort()
        
        if not image_files:
            print("❌ No .png images found!")
            return

        print(f"Processing {len(image_files)} images...")

        for img_name in image_files:
            img_path = os.path.join(IMAGE_FOLDER, img_name)
            img = cv2.imread(img_path)
            if img is None: continue
            
            img_h, img_w = img.shape[:2]
            results = model.predict(img_path, conf=0.35, verbose=False)
            
            if img_name not in positions:
                continue

            drone_pos = positions[img_name]

            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                off_fwd, off_right = pixel_to_meters(center_x, center_y, img_w, img_h, DRONE_ALTITUDE)
                
                human_x = drone_pos['x'] + off_fwd
                human_y = drone_pos['y'] + off_right
                
                print(f"Found in {img_name}: Human at X={human_x:.2f}, Y={human_y:.2f}")
                f_out.write(f"{img_name},{drone_pos['x']:.2f},{drone_pos['y']:.2f},{human_x:.2f},{human_y:.2f}\n")
                found_count += 1

    print("\n--- FINAL REPORT ---")
    if found_count > 0:
        print(f"✅ SUCCESS! Found {found_count} humans.")
        print(f"✅ File saved at: {OUTPUT_FILE}")
    else:
        print("⚠️ Completed, but NO humans were detected.")

if __name__ == "__main__":
    main()
