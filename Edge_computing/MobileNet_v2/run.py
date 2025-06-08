import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image

# 載入標籤檔
with open("imagenet_labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# 初始化 Edge TPU 模型
interpreter = tflite.Interpreter(
    model_path="mobilenet_v2_1.0_224_quant_edgetpu.tflite",
    experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
)
interpreter.allocate_tensors()

# 載入圖片並前處理
img = Image.open("test.jpg").resize((224, 224))
input_data = np.expand_dims(np.array(img).astype(np.uint8), axis=0)

# 推論
input_index = interpreter.get_input_details()[0]['index']
interpreter.set_tensor(input_index, input_data)
interpreter.invoke()

# 取得預測結果
output_index = interpreter.get_output_details()[0]['index']
output = interpreter.get_tensor(output_index)[0]
top_k = output.argsort()[-5:][::-1]

# 顯示前 5 高機率結果
for i in top_k:
    print(f"{labels[i]}: {output[i]/255:.2%}")
