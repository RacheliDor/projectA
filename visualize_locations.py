import os
import math
import cv2
import shutil
from ultralytics import YOLO

# --- הגדרות ---
# ודאי שהנתיבים האלו מדוייקים!
LOG_FILE_PATH = '/home/user_104/RR/drone_full_scan_results/positions_log.txt'
IMAGE_FOLDER = '/home/user_104/RR/drone_full_scan_results/3' 
MODEL_PATH = '/home/user_104/RR/runs/drone_model_final_small/weights/best.pt'

# איפה לשמור את התמונות המצוירות?
OUTPUT_VISUAL_DIR = '/home/user_104/RR/annotated_images'
OUTPUT_CSV_FILE = '/home/user_104/RR/final_human_locations.csv'

# נתוני מצלמה
FOV = 90.0
DRONE_ALTITUDE = 20.0 

def setup_directories():
    if os.path.exists(OUTPUT_VISUAL_DIR):
        shutil.rmtree(OUTPUT_VISUAL_DIR)
    os.makedirs(OUTPUT_VISUAL_DIR)
    print(f"Created output folder at: {OUTPUT_VISUAL_DIR}")

def parse_log_file(log_path):
    coords_db = {}
    if not os.path.exists(log_path):
        print(f"❌ ERROR: Log file NOT found at: {log_path}")
        return {}

    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
            for line in lines[2:]: # Skip headers
                parts = line.split('|')
                if len(parts) >= 6:
                    x = float(parts[2].strip())
                    y = float(parts[3].strip())
                    filename = parts[5].strip() 
                    coords_db[filename] = {'x': x, 'y': y}
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
    
    offset_x_meters = x_dist_px * meters_per_pixel_x
    offset_y_meters = y_dist_px * meters_per_pixel_y
    
    real_offset_forward = -offset_y_meters 
    real_offset_right = offset_x_meters    
    
    return real_offset_forward, real_offset_right

def main():
    print("\n--- STARTING VISUALIZATION ---")
    setup_directories()
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(IMAGE_FOLDER):
        print("❌ Error: Check your paths (Model or Images folder).")
        return
        
    positions = parse_log_file(LOG_FILE_PATH)
    if not positions: return

    print("Loading Model...")
    model = YOLO(MODEL_PATH)
    
    found_count = 0
    
    # פתיחת קובץ ה-CSV לשמירה
    with open(OUTPUT_CSV_FILE, 'w') as f_out:
        f_out.write("Image_File,Drone_X,Drone_Y,Human_Real_X,Human_Real_Y\n")
        
        image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith('.png')]
        image_files.sort()
        
        print(f"Processing {len(image_files)} images...")

        for img_name in image_files:
            img_path = os.path.join(IMAGE_FOLDER, img_name)
            
            # קריאת התמונה באמצעות OpenCV
            img = cv2.imread(img_path)
            if img is None: continue
            
            img_h, img_w = img.shape[:2]
            
            # ביצוע זיהוי
            results = model.predict(img_path, conf=0.35, verbose=False)
            
            if img_name not in positions:
                continue

            drone_pos = positions[img_name]
            detections_in_image = False

            for box in results[0].boxes:
                # קבלת קואורדינטות הריבוע (פיקסלים)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # חישוב מרכז הריבוע
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # חישוב המיקום האמיתי
                off_fwd, off_right = pixel_to_meters(center_x, center_y, img_w, img_h, DRONE_ALTITUDE)
                human_x = drone_pos['x'] + off_fwd
                human_y = drone_pos['y'] + off_right
                
                # --- ציור על התמונה ---
                # 1. ציור הריבוע הירוק
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # 2. הכנת הטקסט עם המיקום
                label_text = f"Person: ({human_x:.1f}, {human_y:.1f})"
                
                # 3. הוספת רקע שחור קטן לטקסט כדי שיהיה קריא
                (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(img, (int(x1), int(y1) - 20), (int(x1) + w, int(y1)), (0, 255, 0), -1)
                
                # 4. כתיבת הטקסט בלבן
                cv2.putText(img, label_text, (int(x1), int(y1) - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # שמירה ל-CSV
                f_out.write(f"{img_name},{drone_pos['x']:.2f},{drone_pos['y']:.2f},{human_x:.2f},{human_y:.2f}\n")
                detections_in_image = True
                found_count += 1
            
            # שמירת התמונה המצוירת רק אם נמצא בה אדם
            if detections_in_image:
                save_path = os.path.join(OUTPUT_VISUAL_DIR, img_name)
                cv2.imwrite(save_path, img)
                print(f"Saved annotated image: {img_name}")

    print("\n--- FINAL REPORT ---")
    if found_count > 0:
        print(f"✅ SUCCESS! Found {found_count} humans.")
        print(f"🖼️  Check the annotated images in: {OUTPUT_VISUAL_DIR}")
        print(f"📄 Data saved to: {OUTPUT_CSV_FILE}")
    else:
        print("⚠️ No humans detected.")

if __name__ == "__main__":
    main()
