水利，低空目标检测数据集的应用  使用yolov8模型训练无人机河道两边地物检测数据集 基于训练好的无人机河道附件目标检测数据集的权重，建立深度学习河道检测系统
以下文字及代码仅供参考学习使用。
@[toc]
**无人机河道两边地物检测数据集
无人机类型：DJI Mavic Air 2
数据类型：图片+目标检测的标签
总图片：508张
图片的分辨率：800*600（数据裁剪）
采集的高度：100m
采集角度：90度
类别：船只，码头，游艇，汽车，摩艇
应用场景：用于水利，偷渡等低空目标检测领域**
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7503cc35c8e847edba542b79de8fef02.png)

使用YOLOv8训练一个基于无人机采集的河道两边地物检测数据集，按照以下步骤进行：数据集准备与预处理、划分数据集、创建`data.yaml`文件、环境搭建、模型训练、推理与部署、性能评估以及模型导出。以下是每个步骤的详细指南和代码示例。仅供参考学习。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9a7d965762124454a9f9d12251a617eb.png)


### 1. 数据集准备与预处理
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7ce70c6ae9824f40861f58d88e0f7a93.png)

假设同学的数据集已经按照以下结构组织：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9b0c02dd4f0f41bfb40f8e68c1f2680c.png)

```
riverbank_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

确保所有的图像文件和对应的YOLO格式标签文件（`.txt`）都已正确命名，并且放置在相应的目录中。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/78931bf84ed1488d9cf7e7ebb5cca445.png)

如果数据集尚未划分，请参考下面的Python脚本将数据集划分为训练集、验证集和测试集（例如7:2:1的比例）：

```python
import os
import random
import shutil

def split_dataset(source_dir, dest_train, dest_val, dest_test, ratio=(0.7, 0.2, 0.1)):
    images = [f for f in os.listdir(os.path.join(source_dir, 'images')) if f.endswith('.jpg')]
    random.shuffle(images)
    
    total = len(images)
    train_size = int(total * ratio[0])
    val_size = int(total * ratio[1])
    
    for i, image_file in enumerate(images):
        label_file = image_file.replace('.jpg', '.txt')
        
        if i < train_size:
            shutil.move(os.path.join(source_dir, 'images', image_file), os.path.join(dest_train, 'images', image_file))
            shutil.move(os.path.join(source_dir, 'labels', label_file), os.path.join(dest_train, 'labels', label_file))
        elif i < train_size + val_size:
            shutil.move(os.path.join(source_dir, 'images', image_file), os.path.join(dest_val, 'images', image_file))
            shutil.move(os.path.join(source_dir, 'labels', label_file), os.path.join(dest_val, 'labels', label_file))
        else:
            shutil.move(os.path.join(source_dir, 'images', image_file), os.path.join(dest_test, 'images', image_file))
            shutil.move(os.path.join(source_dir, 'labels', label_file), os.path.join(dest_test, 'labels', label_file))

# 示例调用
split_dataset('path/to/source/riverbank_dataset', 
              'path/to/dest/riverbank_dataset/train',
              'path/to/dest/riverbank_dataset/val',
              'path/to/dest/riverbank_dataset/test')
```

### 2. 创建 `data.yaml` 文件

在项目根目录下创建一个 `data.yaml` 文件，用于定义数据集路径、类别数量及名称等信息。内容如下：

```yaml
train: ./riverbank_dataset/images/train
val: ./riverbank_dataset/images/val
test: ./riverbank_dataset/images/test

nc: 5 # 类别数量
names: ['boat', 'dock', 'yacht', 'car', 'jet_ski'] # 类别名称
```

### 3. 环境搭建

安装必要的依赖库，特别是Ultralytics YOLOv8：

```bash
pip install ultralytics
```

### 4. 模型训练

加载YOLOv8并使用上面创建的`data.yaml`进行模型训练。下面是一个简单的Python脚本示例：

```python
from ultralytics import YOLO

# 加载YOLOv8模型，可以选择不同尺寸如'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (extra-large)
model = YOLO('yolov8n.yaml')  # 或者直接加载预训练模型例如 'yolov8n.pt'

# 开始训练
results = model.train(data='path/to/data.yaml', epochs=100, imgsz=640, batch=16)

# 注意：可以根据需要调整参数，比如增加epoch数量、修改学习率等
```

### 5. 推理与部署

完成训练后，可以使用模型对新的图像或视频流进行推理。下面是一个简单的推理示例：

```python
import cv2

def detect_riverbank_objects(image_path):
    # 读取图像
    img = cv2.imread(image_path)
    
    # 使用模型进行预测
    results = model(img)  # 返回的结果包含边界框、置信度等信息
    
    for result in results:
        boxes = result.boxes  # 获取所有检测到的对象的边界框
        for box in boxes:
            x1, y1, x2, y2, confidence, class_id = box.xyxy.tolist() + [box.conf.item(), int(box.cls.item())]
            label = model.names[class_id]  # 根据class_id获取类别名称
            
            print(f"Detected {label} at ({x1}, {y1}) - ({x2}, {y2}), Confidence: {confidence}")
            
            # 在原图上绘制边界框和标签
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, f'{label}: {confidence:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    
    # 显示结果图像
    cv2.imshow('Riverbank Object Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

对于实时视频流，你可以类似地读取帧并进行预测：

```python
cap = cv2.VideoCapture('video.mp4')  # 或者使用数字索引如0来捕获摄像头视频流

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 使用模型进行预测
    results = model(frame)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2, confidence, class_id = box.xyxy.tolist() + [box.conf.item(), int(box.cls.item())]
            label = model.names[class_id]
            
            # 绘制边界框和标签
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}: {confidence:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    
    # 显示帧
    cv2.imshow('Riverbank Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 6. 性能评估

使用验证集评估模型性能：

```bash
yolo task=detect mode=val model=path/to/best.pt data=path/to/data.yaml
```

它会输出 mAP、Precision、Recall 等指标。

### 7. 模型导出

如果你想将模型导出为其他格式（如ONNX），以便于部署在不同的环境中，可以使用以下命令：

```bash
yolo export model=path/to/best.pt format=onnx
```

支持的格式包括：`onnx`, `engine(TensorRT)`, `coreml`, `tflite`, `pb(tensorflow)` 等。

### 总结

通过上述步骤，你就可以基于提供的无人机河道两边地物检测数据集使用YOLOv8来训练一个深度学习目标检测模型。这包括了数据集准备与预处理、创建`data.yaml`配置文件、环境搭建、模型训练、推理与部署、性能评估以及模型导出等环节。做个参考

基于训练好的YOLOv8权重文件建立一个无人机河道两边地物检测识别系统，涉及几个关键步骤：加载模型、实现推理逻辑、处理输入数据（如无人机拍摄的图像或视频流）、展示结果以及系统的部署。以下是详细的指南和代码示例。仅供参考和学习。

### 1. 加载训练好的模型

首先确保你的环境中安装了必要的依赖库：

```bash
pip install ultralytics opencv-python
```

然后，使用训练好的权重文件（例如`best.pt`）来加载模型：

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('path/to/your/best.pt')
```

### 2. 实现推理逻辑

编写代码以处理无人机拍摄的图像或视频，并利用模型进行预测。这里提供了一个简单的例子，演示如何对单张图像进行检测：

```python
import cv2

def detect_riverbank_objects(image_path):
    # 读取图像
    img = cv2.imread(image_path)
    
    # 使用模型进行预测
    results = model(img)  # 返回的结果包含边界框、置信度等信息
    
    for result in results:
        boxes = result.boxes  # 获取所有检测到的对象的边界框
        for box in boxes:
            x1, y1, x2, y2, confidence, class_id = box.xyxy.tolist() + [box.conf.item(), int(box.cls.item())]
            label = model.names[class_id]  # 根据class_id获取类别名称
            
            print(f"Detected {label} at ({x1}, {y1}) - ({x2}, {y2}), Confidence: {confidence}")
            
            # 在原图上绘制边界框和标签
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, f'{label}: {confidence:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    
    # 显示结果图像
    cv2.imshow('Riverbank Object Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

对于实时视频流（比如无人机实时传输的视频），可以对上述代码稍作修改，以从摄像头或视频文件中读取帧并进行检测：

```python
cap = cv2.VideoCapture('video.mp4')  # 或者使用数字索引如0来捕获摄像头视频流

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 使用模型进行预测
    results = model(frame)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2, confidence, class_id = box.xyxy.tolist() + [box.conf.item(), int(box.cls.item())]
            label = model.names[class_id]
            
            # 绘制边界框和标签
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}: {confidence:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    
    # 显示帧
    cv2.imshow('Riverbank Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 3. 系统部署

为了将这个检测系统部署到实际应用中，可以根据具体需求选择不同的部署方式：

- **Web服务**：使用Flask或Django等框架创建一个Web服务，通过API接收图片或视频流并返回检测结果。
  
  示例：使用Flask创建一个简单的Web服务
  
  ```python
  from flask import Flask, request, jsonify
  import numpy as np
  from PIL import Image
  import io
  
  app = Flask(__name__)
  
  @app.route('/detect', methods=['POST'])
  def detect():
      if 'image' not in request.files:
          return jsonify({'error': 'No image provided'}), 400
      
      file = request.files['image']
      img_bytes = file.read()
      img = Image.open(io.BytesIO(img_bytes))
      
      # 转换为适合YOLO模型输入的格式
      img_array = np.array(img)
      
      # 使用模型进行预测
      results = model(img_array)
      
      detections = []
      for result in results:
          for box in result.boxes:
              x1, y1, x2, y2, confidence, class_id = box.xyxy.tolist() + [box.conf.item(), int(box.cls.item())]
              detections.append({
                  'bbox': [x1, y1, x2, y2],
                  'confidence': confidence,
                  'class': model.names[class_id]
              })
      
      return jsonify(detections)
  
  if __name__ == '__main__':
      app.run(host='0.0.0.0', port=5000)
  ```

- **边缘设备部署**：如果目标是部署在无人机或其他嵌入式系统上，考虑优化模型以便于在资源受限的设备上运行，或者使用专门的硬件加速器（如NVIDIA Jetson系列）。
- **云端解决方案**：利用云服务提供商（如AWS, Azure）提供的计算资源，构建基于容器（如Docker）的微服务架构。

### 4. 性能评估

使用验证集评估模型性能：

```bash
yolo task=detect mode=val model=path/to/best.pt data=path/to/data.yaml
```

它会输出 mAP、Precision、Recall 等指标。

### 5. 模型导出

如果你想将模型导出为其他格式（如ONNX），以便于部署在不同的环境中，可以使用以下命令：

```bash
yolo export model=path/to/best.pt format=onnx
```

支持的格式包括：`onnx`, `engine(TensorRT)`, `coreml`, `tflite`, `pb(tensorflow)` 等。

### 总结

以上指南展示了如何基于训练好的YOLOv8权重文件建立一个无人机河道两边地物检测识别系统，涵盖了加载模型、实现推理过程、处理不同类型的输入数据以及系统部署的基本方法。

仅供参考学习使用。
