from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, classify
from PIL import Image
import time
import os
from PIL import ImageDraw, ImageFont


# 載入模型與標籤
model_path = 'model_quant.tflite'
label_path = 'labels.txt'
label_name = ["Leopard_cat", "Not_leopard_cat"]

with open(label_path, 'r') as f:
    labels = [line.strip().split(' ', 1)[1] for line in f.readlines()]

interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()

# 圖片資料夾
image_folder = 'test_images'
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# 預測
for image_name in image_files:
    image_path = os.path.join(image_folder, image_name)
    try:
        image = Image.open(image_path).convert('RGB').resize(common.input_size(interpreter))
        common.set_input(interpreter, image)
        interpreter.invoke() #exec inference

        results = classify.get_classes(interpreter, top_k=1) #get result
        label = results[0].id
        score = results[0].score

        print(f'{image_name} => 類別: {label_name[label]}, 分數: {score:.2f}')
        del image  # 主動釋放記憶體
    except Exception as e:
        print(f'❌ {image_name} 處理錯誤: {e}')

