import cv2                      # OpenCV for webcam + drawing UI
import numpy as np              # NumPy for array operations
from tensorflow import keras    # Keras to load Teachable Machine model
import pygame                   # Pygame to play sound (.mp3)
import time                     # Time used for audio cooldown
from collections import deque   # deque stores prediction history for smoothing

# CONSTANT / CONFIGURATION SETTINGS
IMAGE_SIZE = 224                 # Model input size: 224x224 pixels
CONF_THRESHOLD = 0.60            # Minimum confidence to accept detection (60%)

SMOOTHING_WINDOW = 10            # Number of predictions to average for smoother results
STABLE_FRAMES = 5                # Number of consecutive frames required before updating output
AUDIO_COOLDOWN = 5               # Seconds to wait before playing another audio

MODEL_PATH = "keras_model.h5"    # Path for Teachable Machine Keras model
LABELS_PATH = "labels.txt"       # Path for labels file (class names)

SHOW_MODEL_INPUT = False         # If True, show the 224x224 model input window for debugging
# AUDIO MAP (LABEL -> AUDIO FILE)
audio_map = {
    "iloveyou": "iloveyou.mp3",
    "thankyou": "thankyou.mp3",
    "hello": "hello.mp3",
    "good": "good.mp3",
    "please": "please.mp3",
}

# INITIALIZE AUDIO SYSTEM
pygame.mixer.init()              # Start Pygame sound engine
last_audio_time = 0.0            # Stores last time audio played (to apply cooldown)
last_audio_label = None          # Stores last label that played audio

# LOAD MODEL
model = keras.models.load_model(MODEL_PATH, compile=False)

labels = {}
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(" ", 1)       # Split into index + label name
        if len(parts) != 2:
            continue
        idx_str, name = parts
        labels[int(idx_str)] = name.strip()
# PREDICTION SMOOTHING BUFFER
prediction_history = deque(maxlen=SMOOTHING_WINDOW)
# WEBCAM INITIALIZATION
# OpenCV captures camera 0 (default webcam)
cap = cv2.VideoCapture(0)

# UI SETTINGS
WINDOW_NAME = "Teachable Machine - More Accurate Detection"
UI_W, UI_H = 1280, 720           # UI window size

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

# HELPER FUNCTIONS
def center_crop_square(img):
    """Crop the center of the image into a square (needed to avoid stretching)."""
    h, w, _ = img.shape
    size = min(h, w)
    y1 = (h - size) // 2
    x1 = (w - size) // 2
    return img[y1:y1 + size, x1:x1 + size]


def draw_progress_bar(img, x, y, w, h, value, fill_color, bg_color=(60, 60, 60)):
    """Draw a confidence progress bar on the UI panel."""
    value = max(0.0, min(1.0, float(value)))      # clamp value between 0 and 1

    cv2.rectangle(img, (x, y), (x + w, y + h), bg_color, -1)  # background bar

    fill_w = int(w * value)                        # filled width based on confidence
    if fill_w > 0:
        cv2.rectangle(img, (x, y), (x + fill_w, y + h), fill_color, -1)

    cv2.rectangle(img, (x, y), (x + w, y + h), (120, 120, 120), 1)  # border


def resize_keep_aspect(img, target_w, target_h, pad_color=(0, 0, 0)):
    """Resize an image but keep aspect ratio. Adds padding if needed."""
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    interp = cv2.INTER_LINEAR if scale >= 1.0 else cv2.INTER_AREA
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas[:] = pad_color

    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


def put_text_fit(img, text, x, y, max_w, font, base_scale, color, thickness=2):
    """Automatically reduce text size if it is too long for the UI area."""
    scale = base_scale
    while scale > 0.3:
        (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
        if tw <= max_w:
            break
        scale -= 0.05
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    return scale


def draw_camera_title(combined, cam_w, text="SIGN LANGUAGES DETECTOR"):
    """Draw a title bar at the top of the camera view."""
    bar_h = 70
    bg = (12, 12, 12)

    cv2.rectangle(combined, (0, 0), (cam_w, bar_h), bg, -1)
    cv2.rectangle(combined, (0, bar_h - 4), (cam_w, bar_h), (255, 200, 0), -1)

    cv2.putText(combined, text, (22, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255), 2, cv2.LINE_AA)


def draw_card(img, x, y, w, h, bg=(32, 32, 38), border=(60, 60, 70)):
    """Draw a UI card background with border."""
    cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), border, 1)


def display_predictions(frame, avg_preds, final_label, final_confidence):
    """
    This function builds the UI:
    - Left side: camera view
    - Right side: prediction panel (label, confidence, top-3 predictions, audio status)
    """
    global last_audio_time, last_audio_label

    h, w = UI_H, UI_W
    panel_ratio = 0.30
    panel_w = int(w * panel_ratio)
    cam_w = w - panel_w

    scale_ui = max(0.7, min(1.0, panel_w / 360.0))

    # Resize camera frame to fit left UI
    cam_resized = resize_keep_aspect(frame, cam_w, h, pad_color=(10, 10, 10))

    # Create right panel background
    panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
    panel[:] = (20, 20, 20)

    idx_top = int(np.argmax(avg_preds))                 # Most likely class index
    label = final_label                                 # Stable label
    confidence = float(final_confidence)                # Stable confidence
    conf_pct = confidence * 100.0

    ok = confidence > CONF_THRESHOLD and label != "diam"
    accent = (0, 200, 0) if ok else (0, 0, 200)

    now = time.time()
    cooldown_left = max(0.0, AUDIO_COOLDOWN - (now - last_audio_time))
    last_txt = last_audio_label if last_audio_label else "-"

    # PANEL UI CONTENT
    xpad = int(16 * scale_ui) + 10
    y = 28
    card_w = panel_w - xpad * 2

    # Header Card
    card_h = int(115 * scale_ui)
    draw_card(panel, xpad, y, card_w, card_h)

    put_text_fit(panel, "SIGN RECOGNITION", xpad + 14, y + 32,
                 card_w - 28, cv2.FONT_HERSHEY_SIMPLEX,
                 0.72 * scale_ui, (255, 255, 255), 2)

    cv2.putText(panel, label.upper(), (xpad + 14, y + 85),
                cv2.FONT_HERSHEY_SIMPLEX, 1.15 * scale_ui,
                (255, 255, 255), 3, cv2.LINE_AA)

    y += card_h + 18

    # Confidence Card
    card_h = int(118 * scale_ui)
    draw_card(panel, xpad, y, card_w, card_h)

    cv2.putText(panel, f"Confidence: {conf_pct:.1f}%", (xpad + 14, y + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.66 * scale_ui,
                (220, 220, 220), 2, cv2.LINE_AA)

    bar_y = y + 52
    draw_progress_bar(panel, xpad + 14, bar_y, card_w - 28,
                      int(14 * scale_ui), confidence, accent,
                      bg_color=(45, 45, 45))

    status_txt = "CONFIDENT" if ok else "NOT SURE"
    cv2.putText(panel, f"Status: {status_txt}", (xpad + 14, y + 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68 * scale_ui,
                accent, 2, cv2.LINE_AA)

    y += card_h + 18

    # Top predictions Card
    card_h = int(178 * scale_ui)
    draw_card(panel, xpad, y, card_w, card_h)

    cv2.putText(panel, "Top predictions:", (xpad + 14, y + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68 * scale_ui,
                (255, 255, 255), 2, cv2.LINE_AA)

    top_k = 3
    top_indices = np.argsort(avg_preds)[::-1][:top_k]    # Get top 3 prediction indices

    row_y = y + 68
    name_x = xpad + 14
    bar_x = xpad + int(140 * scale_ui)
    pct_x = xpad + card_w - int(52 * scale_ui)
    bar_w = pct_x - bar_x - 10

    for i in top_indices:
        cls_name = labels.get(int(i), f"cls_{int(i)}")
        val = float(avg_preds[i])
        pct = val * 100.0

        is_top = int(i) == idx_top
        name_color = (0, 200, 255) if is_top else (200, 200, 200)
        fill_color = (0, 180, 255) if is_top else (140, 140, 140)

        cls_short = cls_name[:12]

        cv2.putText(panel, cls_short, (name_x, row_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62 * scale_ui,
                    name_color, 2, cv2.LINE_AA)

        draw_progress_bar(panel, bar_x, row_y - int(12 * scale_ui),
                          bar_w, int(12 * scale_ui),
                          val, fill_color,
                          bg_color=(35, 35, 35))

        cv2.putText(panel, f"{pct:.0f}%", (pct_x, row_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58 * scale_ui,
                    (220, 220, 220), 2, cv2.LINE_AA)

        row_y += int(34 * scale_ui)

    y += card_h + 18

    # Audio status card
    card_h = int(102 * scale_ui)
    draw_card(panel, xpad, y, card_w, card_h)

    cv2.putText(panel, "Audio:", (xpad + 14, y + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68 * scale_ui,
                (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(panel, f"Last: {last_txt}", (xpad + 14, y + 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62 * scale_ui,
                (220, 220, 220), 2, cv2.LINE_AA)

    cv2.putText(panel, f"Cooldown: {cooldown_left:.1f}s", (xpad + 14, y + 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62 * scale_ui,
                (220, 220, 220), 2, cv2.LINE_AA)

    # Combine left camera + right panel
    combined = np.hstack([cam_resized, panel])

    # Add title bar on camera side
    draw_camera_title(combined, cam_w, text="SIGN LANGUAGES DETECTOR")

    # Footer instructions
    cv2.putText(combined, "Press Q to quit", (18, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                (255, 255, 255), 2, cv2.LINE_AA)

    return combined, label, confidence


def handle_audio(label, confidence):
    """
    Plays sound if:
    - Confidence > threshold
    - Label is not "diam" (meaning 'no sign')
    - Cooldown time has passed
    - New label is different from last spoken label
    """
    global last_audio_label, last_audio_time

    now = time.time()

    if confidence > CONF_THRESHOLD and label != "diam":

        # Only play if new label and cooldown passed
        if label != last_audio_label and (now - last_audio_time) > AUDIO_COOLDOWN:

            audio_file = audio_map.get(label)

            if audio_file:
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                last_audio_label = label
                last_audio_time = now

# STABILITY VARIABLES
# These help reduce flickering by requiring stable predictions

stable_label = None
stable_count = 0
final_label = "diam"
final_confidence = 0.0

# MAIN LOOP
try:
    while True:

        # 1) Capture a frame from webcam
        ret, frame = cap.read()
        if not ret:
            break

        # 2) Mirror flip the frame (looks natural like mirror)
        frame = cv2.flip(frame, 1)

        # IMAGE PREPROCESSING

        # 3) Crop center square to match model shape expectations
        cropped = center_crop_square(frame)

        # 4) Resize to model input size (224×224)
        img_resized = cv2.resize(cropped, (IMAGE_SIZE, IMAGE_SIZE),
                                 interpolation=cv2.INTER_AREA)

        # 5) Convert from OpenCV BGR to RGB (model expects RGB)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # 6) Normalize pixel values to [-1, 1] because Teachable Machine requires it
        img_array = img_rgb.astype(np.float32) / 127.5 - 1.0

        # 7) Add batch dimension: shape becomes (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)

        # Optional: show what model sees
        if SHOW_MODEL_INPUT:
            preview = cv2.resize(img_resized, (280, 280))
            cv2.imshow("MODEL INPUT (BGR preview)", preview)

        # MODEL PREDICTION

        # 8) Predict sign classes
        preds = model.predict(img_array, verbose=0)[0]

        # 9) Store predictions in buffer for smoothing
        prediction_history.append(preds)

        # 10) Average predictions to reduce random noise
        avg_preds = np.mean(prediction_history, axis=0)

        # 11) Get class with highest probability
        idx = int(np.argmax(avg_preds))
        current_label = labels.get(idx, f"cls_{idx}")
        current_conf = float(avg_preds[idx])

        # STABILITY CHECK

        # 12) If same label continues, increase stable count
        if current_label == stable_label:
            stable_count += 1
        else:
            stable_label = current_label
            stable_count = 1

        # 13) Only update final output after stable frames achieved
        if stable_count >= STABLE_FRAMES:
            final_label = current_label
            final_confidence = current_conf

        # DRAW UI + PLAY AUDIO

        combined, label, confidence = display_predictions(
            frame, avg_preds, final_label, final_confidence
        )

        # 14) Handle audio playback
        handle_audio(label, confidence)

        # 15) Show final UI output
        cv2.imshow(WINDOW_NAME, combined)

        # 16) Quit program when user presses "q"
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # CLEANUP
    cap.release()             # Stop webcam
    cv2.destroyAllWindows()   # Close OpenCV windows
    pygame.mixer.quit()       # Stop audio system