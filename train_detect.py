import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_ROOT = r"F:\PYTHON\视觉基础\dataset\base"
SAVE_WEIGHT = r"F:\PYTHON\视觉基础\text_self_train.pth"
OUT_DIR = r"F:\PYTHON\视觉基础\detect_out"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE = 320
EPOCHS = 8

# 网络结构不变
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

class UnlabeledTextDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.files = []
        # 只保留能正常打开的图片，过滤损坏文件
        for fname in os.listdir(root):
            if not fname.endswith(("jpg", "png", "jpeg")):
                continue
            full_path = os.path.join(root, fname)
            test_img = cv2.imread(full_path)
            if test_img is not None:
                self.files.append(fname)
            else:
                print(f"跳过损坏文件：{fname}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.files[idx])
        img = cv2.imread(path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, pseudo_mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)

        img_tensor = torch.from_numpy(img).permute(2,0,1).float()/255.0
        mask_tensor = torch.from_numpy(pseudo_mask).unsqueeze(0).float()/255.0
        return img_tensor, mask_tensor

def train():
    dataset = UnlabeledTextDataset(IMG_ROOT)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    model = SimpleSegNet().to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        total_loss = 0
        for im, mask in tqdm(loader):
            im, mask = im.to(DEVICE), mask.to(DEVICE)
            pred = model(im)
            loss = loss_fn(pred, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} loss: {total_loss/len(loader):.4f}")
    torch.save(model.state_dict(), SAVE_WEIGHT)
    print("模型训练完成，权重已保存")
    return model

def draw_box(model, img_path, save_path):
    img_raw = cv2.imread(img_path)
    if img_raw is None:
        return
    h_ori, w_ori = img_raw.shape[:2]
    img_res = cv2.resize(img_raw, (IMG_SIZE, IMG_SIZE))
    tensor = torch.from_numpy(img_res).permute(2,0,1).float()/255.0
    tensor = tensor.unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)[0,0]).cpu().numpy()
    prob = cv2.resize(prob, (w_ori, h_ori))
    contours, _ = cv2.findContours((prob>0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x,y,ww,hh = cv2.boundingRect(cnt)
        cv2.rectangle(img_raw, (x,y), (x+ww,y+hh), (0,255,0), 2)
    cv2.imwrite(save_path, img_raw)

if __name__ == "__main__":
    net = train()
    filelist = os.listdir(IMG_ROOT)
    for fname in tqdm(filelist):
        if not fname.endswith(("jpg","png","jpeg")):
            continue
        draw_box(net, os.path.join(IMG_ROOT, fname), os.path.join(OUT_DIR, fname))
    print("全部检测图生成完毕")