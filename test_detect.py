import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 320
WEIGHT_FILE = r"F:\PYTHON\视觉基础\text_self_train.pth"
TEST_IMG_DIR = r"F:\PYTHON\视觉基础\dataset\query"
SAVE_RESULT_DIR = r"F:\PYTHON\视觉基础\query_test_output"
os.makedirs(SAVE_RESULT_DIR, exist_ok=True)

# 和训练代码完全一致的网络
class SimpleSegNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.out = nn.Conv2d(32, 1, 3, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return self.out(x)

def predict_image(model, img_path, save_path):
    # 只用PIL打开图片，不再使用cv2.imread
    img_pil = Image.open(img_path).convert("RGB")
    w_ori, h_ori = img_pil.size
    img_res = img_pil.resize((IMG_SIZE, IMG_SIZE))

    img_np = np.array(img_res)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        prob_map = torch.sigmoid(model(img_tensor)[0, 0]).cpu().numpy()

    prob_map = np.array(Image.fromarray(prob_map).resize((w_ori, h_ori)))
    mask = (prob_map > 0.5).astype(np.uint8)

    # 寻找轮廓并画框
    import cv2
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_draw = np.array(img_pil)
    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        cv2.rectangle(img_draw, (x, y), (x+ww, y+hh), (0, 255, 0), 2)

    Image.fromarray(img_draw).save(save_path)

if __name__ == "__main__":
    net = SimpleSegNet().to(DEVICE)
    net.load_state_dict(torch.load(WEIGHT_FILE, map_location=DEVICE, weights_only=True))
    print("模型加载完成，开始测试推理")

    file_list = os.listdir(TEST_IMG_DIR)
    for fname in tqdm(file_list):
        if not fname.endswith(("jpg", "png", "jpeg")):
            continue
        try:
            predict_image(net,
                          os.path.join(TEST_IMG_DIR, fname),
                          os.path.join(SAVE_RESULT_DIR, fname))
        except Exception as e:
            continue

    print("模型测试结束，预测结果已输出")