import time
import os
from collections import OrderedDict

import cv2
import torch
import torchinfo
import torchvision
import pynvml 
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torchinfo import summary
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from tqdm.notebook import tqdm

# =========================================
# 超参数设置开始
# =========================================
# os.chdir('./insurance')

pd.set_option("display.max_rows", 100)  # 最大行数
pd.set_option("display.max_columns", 500)  # 最大显示列数
pd.set_option('display.width', 200)  # 150，设置打印宽度
pd.set_option("display.precision", 4)  # 浮点型精度

# 检查CUDA是否可用，并据此设置设备
pynvml.nvmlInit()
tmp = [pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(_)).used for _ in range(torch.cuda.device_count())]
gpu_id = tmp.index(min(tmp))
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
print(f"This py file run on the device: {device}.")

# =========================================
# 超参数设置结束
# =========================================

# 定义数据预处理和加载
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
}

# 加载数据集
train_dir = './data/Vehicle_Insurance_Fraud_Classification/train'
test_dir = './data/Vehicle_Insurance_Fraud_Classification/test'

train_data = datasets.ImageFolder(train_dir, data_transforms['train'])
test_data = datasets.ImageFolder(test_dir, data_transforms['test'])
# dir(test_data)
# test_data.class_to_idx
# test_data.imgs

train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=8, )
test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=8, )


# 定义模型
model_resnet18 = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model_resnet18.fc.in_features
num_class = 2
model_resnet18.fc = nn.Linear(num_ftrs, num_class)  # 2 类分类 (Fraud 和 Non-Fraud)
model_resnet18 = model_resnet18.to(device)

torchinfo.summary(model_resnet18, input_data=[torch.randn(12, 3, 256, 400).to(device)],
                  col_names=["input_size", "output_size", "num_params", "params_percent", "mult_adds", ],
                  # depth=20, device='cpu',
                  )
# 冻结参数
for name, parameter in model_resnet18.named_parameters():
    if 'layer4' not in name and 'fc' not in name:
        parameter.requires_grad = False  

# # 将模型复制到多个GPU上
# if torch.cuda.device_count() > 1:
#     model = nn.DataParallel(model).to(device)

# 定义损失函数和优化器
# criterion = nn.BCELoss().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model_resnet18.parameters(), lr=0.001, )
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.005, )

# 训练模型
# train_model(model, criterion, optimizer, num_epochs=25)
# from tqdm.notebook import tqdm
num_epochs = 30
model_resnet18.train()
for epoch in tqdm(range(num_epochs)):
    pass
    # loop = tqdm(train_loader, leave=True)  # 初始化 tqdm 进度条
    # model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到GPU
        labels.logical_not_()
        # # 将一维标签转换为二维的one-hot编码形式
        # labels = torch.cat((1 - labels.float().unsqueeze(-1), labels.float().unsqueeze(-1)), dim=1)
        outputs = model_resnet18(inputs)        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    tqdm.write(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

# 保存和加载模型
model_path = './results/model_resnet18_parameters.pth'
torch.save(model_resnet18.state_dict(), model_path)

# 定义模型
model_resnet18 = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model_resnet18.fc.in_features
num_class = 2
model_resnet18.fc = nn.Linear(num_ftrs, num_class)  

model_resnet18.load_state_dict(torch.load(model_path))
model_resnet18 = model_resnet18.to(device)

# 清理多GPU训练后的环境
# if torch.cuda.device_count() > 1:
#     model_collect = model.module

# 验证模型
# validate_model(model, test_loader)
from sklearn.metrics import classification_report
model_resnet18.eval()
correct = 0
total = 0
list_pred, list_true = [], []
with torch.no_grad():
    for inputs, labels in tqdm(test_loader):
        pass
        labels.logical_not_()
        
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到GPU
        outputs = model_resnet18(inputs)
        # break
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        # labels = labels.float().unsqueeze(1)
        correct += (predicted == labels).sum().item()
        list_pred.extend(predicted.tolist())
        list_true.extend(labels.tolist())
print(f'Accuracy of the model on the test images: {100 * correct / total}%')
print('=' * 60)
report = classification_report(list_true, list_pred)
print(f'Classification Report:\n{report}')
print('=' * 60)
# 计算评估指标
conf_matrix = confusion_matrix(list_true, list_pred)
# 打印评估指标
print(f"Confusion Matrix:\n{conf_matrix}")

# 加载数据集
test_dir = './data/Vehicle_Insurance_Fraud_Classification/test'
test_data2 = datasets.ImageFolder(test_dir, 
                                  transforms.Compose([
                                      transforms.Resize([600,800]), 
                                      transforms.ToTensor(), 
                                  ]), 
                                 )
batch_size = 128
test_loader2 = DataLoader(test_data2, batch_size=batch_size, shuffle=False, )

from sklearn.metrics import classification_report
correct, total = 0, 0
list_pred, list_true = [], []
img_trans = transforms.Compose([
    transforms.Resize(256,antialias=True), 
    # transforms.CenterCrop(224),                                      
    # transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# 1. 加载预训练的模型
model_resnet18.eval()  # 设置为评估模式
# 2. 选择目标层，以ResNet50为例，可以选择model.layer4[-1]作为目标层
target_layer = [model_resnet18.layer4[-1]]
# 3. 初始化GradCAM
cam = GradCAM(
    model=model_resnet18,
    target_layers=target_layer, 
    # use_cuda=torch.cuda.is_available(),
)

for batch_index, (inputs, labels) in tqdm(enumerate(test_loader2),total=len(test_loader2)):
    pass
    # 做变换处理，因为要保持原图，所以加载器里不做处理
    # 反归一化到 [0, 255] 范围
    inputs_uint8 = (inputs * 255).type(torch.uint8)
    # 转换回 PIL 图像
    # inputs_image = Image.fromarray(inputs_uint8[0].permute(1, 2, 0).numpy())
    inputs_image = inputs_uint8.permute(0, 2, 3, 1, ).numpy() # [B,C,H,W] -> [B,H,W,C]
    inputs = img_trans(inputs)
    # break
    # 反转标签
    labels.logical_not_()

    inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到GPU
    outputs = model_resnet18(inputs)
    # break
    _, predicted = torch.max(outputs.data, 1)

    total += labels.size(0)
    correct += (predicted == labels).sum().item()

    list_pred.extend(predicted.tolist())
    list_true.extend(labels.tolist())
    # break
    # 输出激活图
    # break
    for img_index in range(inputs_image.shape[0]):
        pass
        # 激活图
        img = cv2.resize(inputs_image[img_index], (800, 600))
        rgb_img_float = np.float32(img) / 255
        input_tensor = preprocess_image(rgb_img_float, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 这一步会导致[H,W,C] 变成[1,C,H,W]
        
        # 4. 生成CAM
        # 如果您知道感兴趣的类别，可以将target_category设置为该类别的索引
        # 如果不知道，模型将自动选择概率最高的类别
        targets = None
        target_category = cam(input_tensor=input_tensor, targets=targets, )
        cam_image_none = show_cam_on_image(rgb_img_float, target_category.squeeze(),use_rgb=False)

        preds = 0
        targets = [ClassifierOutputTarget(preds)]
        target_category = cam(input_tensor=input_tensor, targets=targets, )
        cam_image_0 = show_cam_on_image(rgb_img_float, target_category.squeeze(),use_rgb=False)

        preds = 1
        targets = [ClassifierOutputTarget(preds)]
        target_category = cam(input_tensor=input_tensor, targets=targets, )
        cam_image_1 = show_cam_on_image(rgb_img_float, target_category.squeeze(),use_rgb=False)


        # Image.fromarray(cam_image)
        # 获取原始图像的尺寸
        res_np = [inputs_image[img_index], cam_image_none, cam_image_0, cam_image_1, ]
        for _ in range(len(res_np)):
            height, width = res_np[_].shape[:2]
            # 计算新的尺寸，等比例缩放1.5倍
            new_width = int(width * 2)
            new_height = int(height * 2)
            # 使用 cv2.resize 函数进行缩放，使用双线性插值
            res_np[_] = cv2.resize(res_np[_], (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        pil_image = Image.fromarray(np.hstack(res_np))
        # # 使用display函数显示图像
        # print(img_class)
        # print(img_path.split('/')[-1].replace('.jpg',''))    
        # display(pil_image)

        # 绘制结果图
        plt.figure(figsize=(18, 12))
        plt.imshow(pil_image)
        plt.axis('off')
        plt.title(f"真实分类：{'Fraud' if labels[img_index] == 1 else 'Non-Fraud'}，预测分类：{'Fraud' if predicted[img_index] == 1 else 'Non-Fraud'}\n\n四幅图分别是：原图、None激活图、Non-Fraud激活图、Fraud激活图",
                  fontproperties='SimSun',)
        # plt.show()
        output_dir = './results/vis_test'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # plt.savefig(f'{output_dir}/img_{batch_index * batch_size + img_index:04d}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/img_{batch_index * batch_size + img_index:04d}.jpg', dpi=300, bbox_inches='tight')
        # plt.savefig(f'{output_dir}/img_{batch_index * batch_size + img_index:04d}.jpg', )
        plt.close()
        # break

print(f'Accuracy of the model on the test images: {100 * correct / total}%')
print('=' * 60)
report = classification_report(list_true, list_pred)
print(f'Classification Report:\n{report}')
print('=' * 60)
# 计算评估指标
conf_matrix = confusion_matrix(list_true, list_pred)
# 打印评估指标
print(f"Confusion Matrix:\n{conf_matrix}")



