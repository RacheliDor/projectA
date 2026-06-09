import airsim
import time
import numpy as np
import os

# --- Critical Settings ---
AREA_SIZE = 80.0   # Scanning Area (80x80 meters)
ALTITUDE = -20.0   # Altitude (Negative is UP in AirSim, so -20m)
SCAN_STEP = 10.0   # Step size between photos
SPEED = 5          # Flight Speed

# Camera Settings
CAMERA_NAME = "0"
IMAGE_TYPE = airsim.ImageType.Scene 

# --- Paths ---
# Using absolute path to be 100% sure where files go
output_dir = os.path.abspath(os.path.join(os.getcwd(), "drone_full_scan_results"))
os.makedirs(output_dir, exist_ok=True)
output_log_file = os.path.join(output_dir, "positions_log.txt")

print(f"DEBUG: Saving all results to: {output_dir}")

captured_data = [] 

# -----------------------------------

def capture_and_log_data(client, scan_point_id):
    """ 
    Captures a compressed PNG image.
    Includes RETRY logic to handle AirSim 'empty response' bugs.
    """
    
    # 1. Get Position
    current_state = client.getMultirotorState()
    target_pos = current_state.kinematics_estimated.position
    
    x_val = target_pos.x_val 
    y_val = target_pos.y_val 
    z_val = target_pos.z_val

    # 2. Capture Image with Retry
    file_name = None
    success = False

    # Try up to 3 times to get a valid image
    for attempt in range(3):
        try:
            responses = client.simGetImages([
                airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.Scene, False, True)
            ])
            
            # Check if we got valid data
            if responses and responses[0].image_data_uint8 and len(responses[0].image_data_uint8) > 0:
                response = responses[0]
                file_name = f"scan_{scan_point_id}.png"
                file_path = os.path.join(output_dir, file_name)
                
                # Save using standard Python write (safer)
                with open(file_path, 'wb') as f:
                    f.write(response.image_data_uint8)
                
                print(f"  -> SUCCESS: Saved {file_name}")
                success = True
                break # Exit the loop, we are done
            else:
                # If data is empty, wait slightly and retry
                time.sleep(0.2)
        
        except Exception as e:
            print(f"  [!] Attempt {attempt+1} failed: {e}")
            time.sleep(0.2)

    if not success:
        print(f"  [X] FAILURE: Camera returned empty data for point {scan_point_id}")
        file_name = "CAPTURE_FAILED"

    # 3. Log Data
    captured_data.append({
        "id": scan_point_id,
        "x": x_val, "y": y_val, "z": z_val, 
        "image_file": file_name,
        "time": time.strftime("%H:%M:%S")
    })

def autonomous_full_scan():
    try:
        client = airsim.MultirotorClient() 
        print("Attempting connection...")
        client.confirmConnection()
        
    except Exception as e:
        print(f"ERROR: Could not connect to AirSim. Error: {e}")
        return

    client.enableApiControl(True)
    client.armDisarm(True)
    
    start_position = client.getMultirotorState().kinematics_estimated.position
    scan_id = 0
    
    # 1. Takeoff to scanning altitude
    print(f"1. Placing drone at altitude: {abs(ALTITUDE)}m...")
    pose = client.simGetVehiclePose()
    pose.position.z = ALTITUDE 
    client.simSetVehiclePose(pose, ignore_collision=True)
    time.sleep(2) 
    
    print(f"2. Starting Grid Scan ({AREA_SIZE}x{AREA_SIZE}m)...")
    
    # ----------------------------------------------------
    # 2. Grid Scan Loop
    # ----------------------------------------------------
    
    for y in np.arange(0, AREA_SIZE + SCAN_STEP, SCAN_STEP):
        
        # Zigzag logic
        if int(y / SCAN_STEP) % 2 == 0:
            x_start, x_end, x_step = 0, AREA_SIZE, SCAN_STEP
        else:
            x_start, x_end, x_step = AREA_SIZE, 0, -SCAN_STEP
        
        # Move Y
        client.moveToPositionAsync(x_start, y, ALTITUDE, SPEED).join()
        
        # Move X
        for x in np.arange(x_start, x_end + x_step if x_step > 0 else x_end - x_step, x_step):
            
            client.moveToPositionAsync(x, y, ALTITUDE, SPEED).join()
            
            scan_id += 1
            print(f"\nScanning point {scan_id} at (X={x:.1f}, Y={y:.1f})...")
            
            # Capture
            capture_and_log_data(client, scan_id)
            time.sleep(0.1) 

    # 3. Return home
    print("\n3. Scanning finished. Returning to origin...")
    client.moveToPositionAsync(start_position.x_val, start_position.y_val, ALTITUDE, SPEED).join() 
    client.landAsync().join()

    # Release control
    client.armDisarm(False)
    client.enableApiControl(False)
    
    # 4. Save Log
    with open(output_log_file, 'w') as f:
        f.write("Scan ID | Time | X | Y | Z | Image File\n")
        f.write("-" * 50 + "\n")
        for item in captured_data:
            f.write(f"{item['id']} | {item['time']} | {item['x']:.2f} | {item['y']:.2f} | {item['z']:.2f} | {item['image_file']}\n")

    print("\n--- MISSION SUMMARY ---")
    print(f"Total scan points completed: {scan_id}")
    print(f"Log saved to: {output_log_file}")
    print(f"Images saved to: {output_dir}")

if __name__ == "__main__":
    autonomous_full_scan()
