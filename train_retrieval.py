import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# ===================== 路径配置（和你的项目保持一致） =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_ROOT = r"F:\PYTHON\视觉基础\dataset\base"
SAVE_WEIGHT = r"F:\PYTHON\视觉基础\text_self_train.pth"
OUT_DIR = r"F:\PYTHON\视觉基础\detect_out"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE = 320
EPOCHS = 8

# ---------------------- 网络结构 ----------------------
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

# ---------------------- 数据集：完全照搬你train.py里的坏图过滤逻辑 ----------------------
class UnlabeledTextDataset(Dataset):
    def __init__(self, root):
        self.img_paths = []
        # 和检索任务一模一样：用PIL预校验文件，过滤破损图片
        for filename in os.listdir(root):
            if filename.lower().endswith(("jpg", "jpeg", "png")):
                full_path = os.path.join(root, filename)
                try:
                    with Image.open(full_path) as test_img:
                        test_img.verify()
                    self.img_paths.append(full_path)
                except (UnidentifiedImageError, OSError):
                    print(f"跳过损坏图片: {filename}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        img_np = cv2.resize(img_np, (IMG_SIZE, IMG_SIZE))

        # 亮度阈值生成伪标签，不需要json标注
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        _, pseudo_mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)

        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(pseudo_mask).unsqueeze(0).float() / 255.0
        return img_tensor, mask_tensor

# ---------------------- 训练流程 ----------------------
def train():
    dataset = UnlabeledTextDataset(IMG_ROOT)
    print(f"筛选后可用训练图片总数：{len(dataset)}")
    if len(dataset) == 0:
        raise RuntimeError("没有可用图片，请检查文件夹")

    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    model = SimpleSegNet().to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        total_loss = 0
        for im, mask in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
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

# ---------------------- 推理画框 ----------------------
def draw_box(model, img_path, save_path):
    try:
        img_raw = cv2.imread(img_path)
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
    except:
        return

if __name__ == "__main__":
    net = train()
    # 批量生成检测图
    valid_paths = []
    for fname in os.listdir(IMG_ROOT):
        if fname.lower().endswith(("jpg","png","jpeg")):
            fp = os.path.join(IMG_ROOT, fname)
            try:
                with Image.open(fp) as im:
                    im.verify()
                valid_paths.append((fp, fname))
            except:
                continue
    for img_path, fname in tqdm(valid_paths):
        draw_box(net, img_path, os.path.join(OUT_DIR, fname))
    print("全部文字检测效果图生成完毕")