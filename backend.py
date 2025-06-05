import threading
import subprocess
import time
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import cv2
from scipy.io import wavfile
from ultralytics import YOLO
from collections import deque
import queue


# === CONFIGURATION (mostly same) ===
VIDEO_PATH = '/Users/veeshal/Downloads/Video_Request_Fence_Approach.mp4
YOLO_MODEL_PATH = "/Users/veeshal/Downloads/best.pt"
WAV_OUTPUT_PATH = "audio.wav"

YOLO_CONFIDENCE_THRESHOLD = 0.45
YOLO_IOU_THRESHOLD = 0.7
FRAME_CHANGE_THRESHOLD_PERCENT = 3
# SHORT_TERM_BUFFER_SIZE = 5 # Less critical now, but can still be used for micro-confirmations

CLASS_NAME_MAP = {
    'civilian': 'civilian', 'soldier': 'soldier', 'weapon': 'weapon', 'fence': 'fence',
}
CLASS_CIVILIAN = CLASS_NAME_MAP.get('civilian', 'civilian')
CLASS_SOLDIER = CLASS_NAME_MAP.get('soldier', 'soldier')
CLASS_WEAPON = CLASS_NAME_MAP.get('weapon', 'weapon')
CLASS_FENCE = CLASS_NAME_MAP.get('fence', 'fence')

WEAPON_PROXIMITY_THRESHOLD = 75

YAMNET_TARGET_SOUNDS = {
    "threat": ["Gunshot, gunfire", "Explosion", "Machine gun", "Artillery fire"],
    "fence_tampering": ["Squeak", "Scrape", "Metal", "Breaking", "Crowbar", "Hammer","Tick","Scissors","Glass","Tick-tock","Coin(dropping)","Whip","Finger Snapping","Mechanisms"],
    "ambient": ["Speech", "Vehicle"], "alert": ["Alarm", "Siren"]
}
YAMNET_CONFIDENCE_THRESHOLD = 0.25

# --- Global Shared Data Structures ---
# For enhanced_viewer (all frames with any detection)
yolo_frames_with_detections_buffer_for_viewer = [] # Renamed for clarity
# Stores all processed frames' basic data if needed for Qwen later
all_processed_frames_metadata = []

# --- Event Aggregation Data ---
event_evidence = {
    "explosion_sound_detected": False,
    "gunfire_sound_detected": False,
    "fence_tampering_sound_detected": False,
    "civilian_with_weapon_frames": 0, # Count of frames where this occurs
    "civilian_near_fence_with_sound_frames": 0,
    "unarmed_civilian_frames": 0,
    "soldier_activity_frames": 0,
    "first_significant_event_frame_id": -1, # Frame ID where the *primary* event is considered to start
    "primary_event_type_for_qwen": None # Will be set after all processing
}
# YAMNet overall results
yamnet_overall_results = {"detected_sounds": [], "highest_confidence_sound": ("None", 0.0)}

# For real-time display
display_queue = queue.Queue(maxsize=60) # Increased slightly
stop_event = threading.Event()

# === HELPER FUNCTIONS (Identical) ===
def get_center_bbox(bbox): x1, y1, x2, y2 = bbox; return int((x1 + x2) / 2), int((y1 + y2) / 2)
def calculate_distance_bboxes(bbox1, bbox2):
    c1, c2 = get_center_bbox(bbox1), get_center_bbox(bbox2)
    return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

# === AUDIO THREAD: YAMNet Logic (Modified to update event_evidence directly for sounds) ===
def run_yamnet():
    global event_evidence # To update sound detection flags
    print("🔊 [YAMNet] Extracting audio...")
    try:
        cmd = f'ffmpeg -i "{VIDEO_PATH}" -ac 1 -ar 16000 -sample_fmt s16 "{WAV_OUTPUT_PATH}" -y -hide_banner -loglevel error'
        subprocess.run(cmd, shell=True, check=True, timeout=60)
    except Exception as e:
        print(f"🔊 [YAMNet] ERROR: ffmpeg failed: {e}"); return

    print("🔊 [YAMNet] Running audio classification...")
    try:
        model_audio = hub.load('https://tfhub.dev/google/yamnet/1')
        class_map_path = model_audio.class_map_path().numpy()
        class_names_yamnet = [row['display_name'] for row in csv.DictReader(tf.io.gfile.GFile(class_map_path))]
        
        sample_rate, wav_data = wavfile.read(WAV_OUTPUT_PATH)
        waveform = wav_data / (tf.int16.max if wav_data.dtype == np.int16 else np.iinfo(wav_data.dtype).max)
        if len(waveform.shape) > 1: waveform = np.mean(waveform, axis=1)

        scores, _, _ = model_audio(waveform)
        scores_np = scores.numpy()

        detected_sounds_this_file = []
        max_conf = 0.0; best_sound = "None"

        for i, class_name in enumerate(class_names_yamnet):
            max_score_for_class = np.max(scores_np[:, i])
            if max_score_for_class >= YAMNET_CONFIDENCE_THRESHOLD:
                detected_sounds_this_file.append((class_name, float(max_score_for_class)))
                if max_score_for_class > max_conf: max_conf, best_sound = max_score_for_class, class_name
                
                # Update global event_evidence based on YAMNet categories
                if class_name in YAMNET_TARGET_SOUNDS["threat"]:
                    if "Explosion" in class_name or "Artillery" in class_name:
                        event_evidence["explosion_sound_detected"] = True
                    elif "Gunshot" in class_name or "Machine gun" in class_name:
                        event_evidence["gunfire_sound_detected"] = True
                if class_name in YAMNET_TARGET_SOUNDS["fence_tampering"]:
                    event_evidence["fence_tampering_sound_detected"] = True
        
        yamnet_overall_results["detected_sounds"] = sorted(detected_sounds_this_file, key=lambda x: x[1], reverse=True)
        yamnet_overall_results["highest_confidence_sound"] = (best_sound, float(max_conf))

        if detected_sounds_this_file: print(f"🔊 [YAMNet] Detected: {yamnet_overall_results['detected_sounds']}")
        else: print("🔊 [YAMNet] No relevant sounds detected above threshold.")
    except Exception as e: print(f"🔊 [YAMNet] ERROR in classification: {e}")


# === PER-FRAME ANALYSIS TO UPDATE EVIDENCE (not deciding final event here) ===
def update_event_evidence_from_frame(current_frame_data, yolo_class_names_dict_ref):
    """Updates global event_evidence based on current frame. Doesn't decide the final event."""
    global event_evidence
    detections = current_frame_data['detections']
    frame_id = current_frame_data['frame_id']

    civilians = [d for d in detections if d['class_name'] == CLASS_CIVILIAN]
    soldiers = [d for d in detections if d['class_name'] == CLASS_SOLDIER]
    weapons = [d for d in detections if d['class_name'] == CLASS_WEAPON]

    #if civilians :
     #   print(f"🔍 [Event Check] Frame {frame_id}: Found civilians and fences. Checking for tampering...")

    # --- Civilian with Weapon ---
    is_civ_with_weapon_this_frame = False
    for civ in civilians:
        for wep in weapons:
            if calculate_distance_bboxes(civ['bbox'], wep['bbox']) < WEAPON_PROXIMITY_THRESHOLD:
                is_civ_with_weapon_this_frame = True
                break
        if is_civ_with_weapon_this_frame: break
    
    if is_civ_with_weapon_this_frame:
        event_evidence["civilian_with_weapon_frames"] += 1
        return "civilian_with_weapon" # Potential event type this frame

    # --- Civilian near Fence (check YAMNet's global flag later) ---
    if civilians and event_evidence["fence_tampering_sound_detected"]: # Check global sound flag
        for civ in civilians:
            event_evidence["civilian_near_fence_with_sound_frames"] += 1
            return "civilian_fence_tamper" # Potential event type

    # --- Unarmed Civilian Presence ---
    if civilians and not is_civ_with_weapon_this_frame:
        event_evidence["unarmed_civilian_frames"] += 1
        return "unarmed_civilian" # Potential event type

    # --- Soldier Activity ---
    if soldiers and not civilians: # Assuming soldiers are "normal" if no civilians are causing issues
        event_evidence["soldier_activity_frames"] += 1
        return "soldier_activity" # Potential event type
    
    return "other" # No specific tracked event category this frame


# === VIDEO THREAD: YOLO Video Processing (Modified for evidence aggregation) ===
def run_yolo_video(stop_event_ref, display_queue_ref):
    global yolo_frames_with_detections_buffer_for_viewer, all_processed_frames_metadata, event_evidence

    print("🎥 [YOLO] Initializing YOLO model...")
    try:
        model_yolo = YOLO(YOLO_MODEL_PATH)
        yolo_class_names_dict = model_yolo.names
    except Exception as e:
        print(f"🎥 [YOLO] ERROR: Could not load YOLO model: {e}")
        display_queue_ref.put(None); return

    print(f"🎥 [YOLO] Opening video file: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"🎥 [YOLO] ERROR: Could not open video file."); display_queue_ref.put(None); return

    frame_id_counter = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 25.0 

    prev_gray_frame_eq = None
    # short_term_frame_history = deque(maxlen=SHORT_TERM_BUFFER_SIZE) # Can still be useful for micro-logic if needed

    print("🎥 [YOLO] Starting video processing...")
    while cap.isOpened():
        if stop_event_ref.is_set(): print("🎥 [YOLO] Stop event received."); break
        success, frame_bgr = cap.read()
        if not success: break

        # Frame Differencing (same)
        gray_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_frame_eq = cv2.equalizeHist(gray_frame)
        if prev_gray_frame_eq is not None:
            frame_diff = cv2.absdiff(gray_frame_eq, prev_gray_frame_eq)
            if frame_diff.size > 0:
                diff_score = np.sum(frame_diff) / (frame_diff.size * 255) * 100
                if diff_score < FRAME_CHANGE_THRESHOLD_PERCENT:
                    frame_id_counter += 1
                    prev_gray_frame_eq = gray_frame_eq.copy()
                    try: display_queue_ref.put(frame_bgr.copy(), timeout=0.01)
                    except queue.Full: pass
                    continue
        prev_gray_frame_eq = gray_frame_eq.copy()

        results = model_yolo.predict(frame_bgr, save=False, iou=YOLO_IOU_THRESHOLD, conf=YOLO_CONFIDENCE_THRESHOLD, verbose=False)
        timestamp_sec = frame_id_counter / fps
        time_str = time.strftime('%H:%M:%S', time.gmtime(timestamp_sec)) + f".{int((timestamp_sec % 1) * 1000):03d}"
        
        current_frame_detections_list = []
        display_frame_for_queue = frame_bgr.copy()

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_id = int(box.cls[0].item())
                prob = round(box.conf[0].item(), 2)
                class_name_detected = yolo_class_names_dict[class_id]
                
                current_frame_detections_list.append({
                    'bbox': [x1, y1, x2, y2], 'class_id': class_id, 
                    'class_name': class_name_detected, 'confidence': prob
                })
                label = f"{class_name_detected}: {prob:.2f}"
                cv2.rectangle(display_frame_for_queue, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame_for_queue, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2)
        
        cv2.putText(display_frame_for_queue, f"T:{time_str} F:{frame_id_counter}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
        try: display_queue_ref.put(display_frame_for_queue, timeout=0.01)
        except queue.Full: pass

        # Store metadata for ALL processed (non-skipped) frames
        # We'll store the frame_bgr itself here. If memory becomes an issue for very long videos,
        # we might need to write frames to temp files and store paths.
        current_frame_full_data = {
            'frame_id': frame_id_counter,
            'timestamp': time_str,
            'frame_bgr': frame_bgr.copy(), # Store the actual frame
            'detections': current_frame_detections_list
        }
        all_processed_frames_metadata.append(current_frame_full_data)

        if current_frame_detections_list: # Only update evidence if there are detections
            # Add to viewer buffer
            yolo_frames_with_detections_buffer_for_viewer.append({
                'frame_id': frame_id_counter, 'timestamp': time_str,
                'frame': frame_bgr.copy(), 'detections': current_frame_detections_list
            })
            
            # Update aggregate event evidence
            # Pass the full data for analysis, which includes the BGR frame if needed for future complex analysis
            potential_event_this_frame = update_event_evidence_from_frame(current_frame_full_data, yolo_class_names_dict)
            
            # If this is the first time a "significant" type of event is seen, mark its frame_id.
            # This is a heuristic for the *start* of the primary event.
            # The definition of "significant" needs to align with your priority for Qwen.
            if event_evidence["first_significant_event_frame_id"] == -1:
                if potential_event_this_frame in ["civilian_with_weapon", "civilian_fence_tamper"] or \
                   event_evidence["explosion_sound_detected"] or event_evidence["gunfire_sound_detected"]:
                    event_evidence["first_significant_event_frame_id"] = frame_id_counter
        
        frame_id_counter += 1
        if frame_id_counter % 100 == 0: print(f"🎥 [YOLO] Processed frame {frame_id_counter}")

    cap.release()
    display_queue_ref.put(None) # Signal end
    print("🎥 [YOLO] Finished video processing.")

# === FUNCTION TO DETERMINE PRIMARY EVENT AND PREPARE QWEN DATA ===
def determine_primary_event_and_prepare_qwen_data():
    global event_evidence, all_processed_frames_metadata
    
    primary_event_type = "No Significant Event"
    qwen_prompt = "General analysis request for the video."
    frames_for_qwen = []
    # How many frames to count an event as "sustained" enough
    SUSTAINED_EVENT_THRESHOLD_FRAMES = 10 # e.g., 10 frames of civ_with_weapon

    # --- Prioritize Events ---
    # 1. Explosion / Gunfire (Audio Driven, Highest Priority)
    if event_evidence["explosion_sound_detected"]:
        primary_event_type = "Explosion Event"
        qwen_prompt = "CRITICAL ALERT: Explosion sound detected. Analyze subsequent video frames for visual confirmation, damage, casualties, and ongoing threats."
        # If first_significant_event_frame_id was set by this, use it, else default to start
        start_frame_id_for_qwen = event_evidence["first_significant_event_frame_id"] if event_evidence["first_significant_event_frame_id"] != -1 else 0
    elif event_evidence["gunfire_sound_detected"]:
        primary_event_type = "Gunfire Event"
        qwen_prompt = "CRITICAL ALERT: Gunfire sound detected. Analyze subsequent video frames for shooters, targets, casualties, and tactical situation."
        start_frame_id_for_qwen = event_evidence["first_significant_event_frame_id"] if event_evidence["first_significant_event_frame_id"] != -1 else 0
    
    # 2. Civilian with Weapon (Visual, sustained)
    elif event_evidence["civilian_with_weapon_frames"] >= SUSTAINED_EVENT_THRESHOLD_FRAMES:
        primary_event_type = "Civilian with Weapon"
        qwen_prompt = f"POTENTIAL THREAT: Civilian observed with a weapon for a sustained period ({event_evidence['civilian_with_weapon_frames']} frames). Analyze actions, weapon type, intent, and surrounding context from the point of first detection onwards."
        start_frame_id_for_qwen = event_evidence["first_significant_event_frame_id"]
        
    # 3. Fence Breach Attempt (Visual + Audio, sustained)
    elif event_evidence["civilian_near_fence_with_sound_frames"] >= SUSTAINED_EVENT_THRESHOLD_FRAMES and \
         event_evidence["fence_tampering_sound_detected"]:
        primary_event_type = "Fence Breach Attempt"
        qwen_prompt = f"SECURITY ALERT: Civilian activity near fence detected for {event_evidence['civilian_near_fence_with_sound_frames']} frames, with concurrent audio suggesting fence tampering. Analyze interaction with the fence and assess breach attempt from the point of first detection onwards."
        start_frame_id_for_qwen = event_evidence["first_significant_event_frame_id"]

    # 4. Unarmed Civilian Presence (Sustained, if no higher threats)
    elif event_evidence["unarmed_civilian_frames"] >= SUSTAINED_EVENT_THRESHOLD_FRAMES * 2: # Higher threshold for "just civilian"
        primary_event_type = "Civilian Presence (Unarmed)"
        qwen_prompt = f"INFO: Sustained civilian presence observed ({event_evidence['unarmed_civilian_frames']} frames) without obvious weapons. Monitor activity, assess intentions, and identify any unusual behavior from the point of first detection onwards."
        start_frame_id_for_qwen = event_evidence["first_significant_event_frame_id"]
        if start_frame_id_for_qwen == -1 and event_evidence["unarmed_civilian_frames"] > 0: # Find first unarmed civ frame if not set
            for idx, frame_meta in enumerate(all_processed_frames_metadata):
                if any(d['class_name'] == CLASS_CIVILIAN for d in frame_meta['detections']):
                    is_armed = False
                    for civ_d in (d for d in frame_meta['detections'] if d['class_name'] == CLASS_CIVILIAN):
                        for wep_d in (d for d in frame_meta['detections'] if d['class_name'] == CLASS_WEAPON):
                            if calculate_distance_bboxes(civ_d['bbox'], wep_d['bbox']) < WEAPON_PROXIMITY_THRESHOLD:
                                is_armed = True; break
                        if is_armed: break
                    if not is_armed:
                        start_frame_id_for_qwen = frame_meta['frame_id']; break


    # 5. Routine Soldier Activity (Default if nothing else significant)
    elif event_evidence["soldier_activity_frames"] > 0:
        primary_event_type = "Routine Soldier Activity"
        qwen_prompt = "INFO: Predominantly soldier activity observed. Provide a general summary of the activities."
        # For routine, maybe send a few keyframes or just first few.
        start_frame_id_for_qwen = 0 # from beginning or specific keyframes
    else:
        # Default if truly nothing specific (e.g. empty video or only unclassified objects)
        primary_event_type = "No Specific Classified Event"
        qwen_prompt = "The video has been processed. No specific pre-defined critical events were detected based on current rules. Perform a general analysis if desired."
        start_frame_id_for_qwen = 0


    # --- Collect frames for Qwen ---
    # Send all frames from 'start_frame_id_for_qwen' to the end of the video.
    if start_frame_id_for_qwen != -1:
        print(f"Primary event '{primary_event_type}' starts around frame {start_frame_id_for_qwen}. Collecting subsequent frames for Qwen.")
        for frame_meta in all_processed_frames_metadata:
            if frame_meta['frame_id'] >= start_frame_id_for_qwen:
                if frame_meta['frame_bgr'] is not None:
                    frames_for_qwen.append({
                        'frame_id': frame_meta['frame_id'],
                        'timestamp': frame_meta['timestamp'],
                        'frame_image_bytes': cv2.imencode('.jpg', frame_meta['frame_bgr'])[1].tobytes(),
                        'detections': frame_meta['detections']
                    })
    
    # If no specific start_frame_id was set by an event, but we have a type, send first few frames as sample
    if not frames_for_qwen and primary_event_type not in ["No Significant Event", "No Specific Classified Event"]:
        print(f"No specific start frame for '{primary_event_type}', sending first 5 detected frames as sample.")
        count = 0
        for frame_meta in all_processed_frames_metadata:
            if frame_meta['frame_bgr'] is not None and frame_meta['detections']: # Send frames with detections
                frames_for_qwen.append({
                    'frame_id': frame_meta['frame_id'], 'timestamp': frame_meta['timestamp'],
                    'frame_image_bytes': cv2.imencode('.jpg', frame_meta['frame_bgr'])[1].tobytes(),
                    'detections': frame_meta['detections']
                })
                count += 1
                if count >= 5: break # Send up to 5 sample frames

    event_evidence["primary_event_type_for_qwen"] = primary_event_type # Store determined type

    if not frames_for_qwen and primary_event_type not in ["No Significant Event", "No Specific Classified Event"]:
         print(f"Warning: Event '{primary_event_type}' determined, but no frames were collected for Qwen.")


    return primary_event_type, qwen_prompt, frames_for_qwen


# === Enhanced Viewer (Identical) ===
def enhanced_viewer(buffer_to_view): # Expects yolo_frames_with_detections_buffer_for_viewer
    if not buffer_to_view: print("👓 [Viewer] Buffer empty."); return
    current_idx = 0
    cv2.namedWindow('Enhanced Detection Viewer', cv2.WINDOW_AUTOSIZE)
    while 0 <= current_idx < len(buffer_to_view):
        d = buffer_to_view[current_idx]
        if 'frame' not in d or d['frame'] is None: current_idx = (current_idx + 1)%len(buffer_to_view); continue
        disp = d['frame'].copy()
        for box in d['detections']:
            x1,y1,x2,y2=map(int,box['bbox'])
            cv2.rectangle(disp,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(disp,f"{box['class_name']}:{box['confidence']:.2f}",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        cv2.putText(disp,f"F:{d['frame_id']} T:{d['timestamp']} Idx:{current_idx+1}/{len(buffer_to_view)}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        cv2.imshow('Enhanced Detection Viewer', disp)
        key=cv2.waitKey(0)&0xFF
        if key==ord('q'):break
        elif key==ord('n'):current_idx=min(current_idx+1,len(buffer_to_view)-1)
        elif key==ord('p'):current_idx=max(current_idx-1,0)
    cv2.destroyAllWindows()
    print("👓 [Viewer] Closed.")


# === MAIN EXECUTION ===
if __name__ == "__main__":
    main_start_time = time.time()

    yamnet_thread = threading.Thread(target=run_yamnet)
    print("🚀 Starting YAMNet audio processing thread...")
    yamnet_thread.start()

    yolo_thread = threading.Thread(target=run_yolo_video, args=(stop_event, display_queue))
    print("🚀 Starting YOLO video processing thread...")
    yolo_thread.start()

    cv2.namedWindow("YOLO Real-time Detections", cv2.WINDOW_AUTOSIZE)
    print("📺 Real-time display started. Press 'q' in window to quit early.")
    while True:
        try:
            frame_to_display = display_queue.get(timeout=0.1)
            if frame_to_display is None: print("📺 YOLO processing finished."); break
            cv2.imshow("YOLO Real-time Detections", frame_to_display)
        except queue.Empty:
            if not yolo_thread.is_alive() and display_queue.empty(): print("📺 YOLO done, queue empty."); break
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): print("📺 'q' pressed, stopping..."); stop_event.set(); break
    cv2.destroyAllWindows()
    print("📺 Real-time display loop ended.")

    print("⏳ Waiting for YAMNet thread..."); yamnet_thread.join(); print("🟢 YAMNet finished.")
    print("⏳ Waiting for YOLO thread..."); yolo_thread.join(timeout=5 if stop_event.is_set() else None)
    if yolo_thread.is_alive(): print("⚠️ YOLO thread did not terminate gracefully.")
    else: print("🟢 YOLO finished.")

    # --- Determine the SINGLE primary event for the video ---
    print("\n🔬 Determining primary event for the video...")
    primary_event, qwen_prompt, qwen_frames = determine_primary_event_and_prepare_qwen_data()
    
    print(f"\n✅ Overall Video Analysis Complete in {time.time() - main_start_time:.2f} seconds.")
    print(f"🔊 YAMNet Sounds: {yamnet_overall_results['detected_sounds']}")
    print(f"📊 Event Evidence Counts: {event_evidence}")

    print(f"\n🤖 === Qwen Submission Plan ===")
    print(f"Primary Event Determined: {primary_event}")
    print(f"Qwen Prompt: {qwen_prompt}")
    if qwen_frames:
        print(f"Frames to send to Qwen: {len(qwen_frames)}")
        print(f"  - From Frame ID: {qwen_frames[0]['frame_id']} ({qwen_frames[0]['timestamp']})")
        print(f"  - To Frame ID:   {qwen_frames[-1]['frame_id']} ({qwen_frames[-1]['timestamp']})")
        # Here you would make the actual API call: call_qwen_api(qwen_prompt, qwen_frames)
    else:
        print("No specific frames identified for Qwen for this event type or no frames matched criteria.")

    print(f"\n📦 Viewer Buffer: {len(yolo_frames_with_detections_buffer_for_viewer)} frames with detections.")
    if yolo_frames_with_detections_buffer_for_viewer:
        print("Launching Enhanced Viewer...")
        enhanced_viewer(yolo_frames_with_detections_buffer_for_viewer)
    else:
        print("ℹ️ No frames for viewer.")
    
    print(f"\n🏁 Pipeline finished. Total time: {time.time() - main_start_time:.2f}s")
