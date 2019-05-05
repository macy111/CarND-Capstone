from yolo import YOLO
import PIL.Image as Image
import numpy as np
from PIL import ImageFile
from matplotlib import pyplot as plt
ImageFile.LOAD_TRUNCATED_IMAGES = True

model = YOLO(model_path="./trained_weights.h5")
lines = ["./frame0010.jpg"]
for p in lines:
    print(p)
    img = Image.open(p).convert('RGB')
    _ , _, pred_classes = model.detect_image(img)
    print(pred_classes)
    plt.imshow(img)
    plt.show()
