import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

labels = [line.strip() for line in open("labels.txt")]
# 印出 labels 看看順序
print(labels)


with open("labels.txt") as f:
    labels = [line.strip().split(' ', 1)[1] for line in f]  # 只取 "Leopard_cat"


TARGET_LABEL_INDEX = labels.index("Leopard_cat")

# 參數

VIDEO_PATH = "test.mp4"
TFLITE_PATH = "model_quant.tflite"
CONF_THRESHOLD = 0.95
INPUT_SIZE = 224



# 初始化 TFLite
interpreter = tflite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 開啟影片
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

frame_index = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 預處理
    resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    input_tensor = np.expand_dims(rgb, axis=0).astype(np.uint8)

    # 推論
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    pred_id = np.argmax(output)
    conf = np.max(output)/ 255.0

    # 偵測到石虎
    if pred_id == TARGET_LABEL_INDEX and conf > CONF_THRESHOLD:
        timestamp_sec = frame_index / fps
        mins = int(timestamp_sec // 60)
        secs = int(timestamp_sec % 60)
        print("Raw output:", output)
        print("Pred id:", pred_id, "Confidence:", conf)

        print(f"⏱️ {mins:02d}:{secs:02d} ➜ 出現了石虎！（信心值：{conf:.2f}）")

    frame_index += 1

cap.release()
print("✅ 偵測完成")
