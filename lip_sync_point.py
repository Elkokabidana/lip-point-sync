import cv2
import mediapipe as mp
import numpy as np
import pyaudio
import threading
import queue
from scipy.signal import find_peaks

# تنظیمات دوربین
WEBCAM_WIDTH = 640
WEBCAM_HEIGHT = 480

# تنظیمات ضبط صوت
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 512  # کاهش اندازه بافر برای کاهش تأخیر

# صف برای ذخیره داده‌های صوتی
audio_queue = queue.Queue()

# تنظیمات MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# نقاط کلیدی برای لب‌ها
UPPER_LIP_POINTS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]  # لب بالا
LOWER_LIP_POINTS = [146, 91, 181, 84, 17, 314, 405, 321, 375, 312]    # لب پایین

# تابع ضبط صوت
def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("🎤 ضبط صوت آغاز شد...")
    
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_queue.put(data)

# تابع استخراج ویژگی‌های صوتی
def extract_audio_features(audio_data):
    # تبدیل داده‌های صوتی به numpy array
    audio_signal = np.frombuffer(audio_data, dtype=np.int16)
    
    # محاسبه طیف‌فرکانسی
    spectrum = np.abs(np.fft.fft(audio_signal))
    frequencies = np.fft.fftfreq(len(spectrum), 1 / RATE)
    
    # پیدا کردن فرکانس‌های اصلی صدا
    peaks, _ = find_peaks(spectrum, height=np.max(spectrum) * 0.5)
    
    if len(peaks) > 0:
        main_freq = np.abs(frequencies[peaks[0]])  # فرکانس غالب
    else:
        main_freq = 0
    
    return main_freq

# تابع پردازش ویدیو
def process_video():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)
    print("📸 پردازش تصویر آغاز شد...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                upper_lip_points = []
                lower_lip_points = []

                # استخراج نقاط کلیدی لب بالا و پایین
                for idx in UPPER_LIP_POINTS:
                    x = int(face_landmarks.landmark[idx].x * WEBCAM_WIDTH)
                    y = int(face_landmarks.landmark[idx].y * WEBCAM_HEIGHT)
                    upper_lip_points.append((x, y))

                for idx in LOWER_LIP_POINTS:
                    x = int(face_landmarks.landmark[idx].x * WEBCAM_WIDTH)
                    y = int(face_landmarks.landmark[idx].y * WEBCAM_HEIGHT)
                    lower_lip_points.append((x, y))

                # همگام‌سازی با صدا
                if not audio_queue.empty():
                    audio_data = audio_queue.get()
                    main_freq = extract_audio_features(audio_data)

                    # تغییر اندازه لب‌ها بر اساس فرکانس
                    lip_height = int(2 + (main_freq / 1000) * 15) if main_freq > 0 else 0
                    
                    # تغییر مستقیم مختصات لب‌ها برای حرکت طبیعی
                    for i in range(len(lower_lip_points)):
                        x, y = lower_lip_points[i]
                        lower_lip_points[i] = (x, min(y + lip_height, WEBCAM_HEIGHT - 1))  # حرکت به سمت پایین

                    for i in range(len(upper_lip_points)):
                        x, y = upper_lip_points[i]
                        upper_lip_points[i] = (x, max(y - lip_height, 0))  # حرکت به سمت بالا

                # رسم لب‌ها با رنگ مشخص
                cv2.polylines(frame, [np.array(upper_lip_points)], isClosed=True, color=(255, 255, 255), thickness=2)
                cv2.polylines(frame, [np.array(lower_lip_points)], isClosed=True, color=(255, 255, 255), thickness=2)

        # نمایش ویدیو
        cv2.imshow('Lip Sync', frame)

        # خروج با فشار دادن 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# اجرای ضبط صوت در یک Thread
audio_thread = threading.Thread(target=record_audio, daemon=True)
audio_thread.start()

# اجرای پردازش ویدیو
process_video()

