import os
import cv2
import numpy as np
import random
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# 1. Configurações
dataset_dir = 'imagens'  # subpastas 'cats', 'dogs'
img_size = (128, 128)
limit_per_class = 300
batch_size = 32
num_epochs = 30
learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Coleta paths e labels limitados e filtra imagens corrompidas
all_paths, all_labels = [], []
for label, cls in enumerate(['cats', 'dogs']):
    cls_dir = os.path.join(dataset_dir, cls)
    files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
             if os.path.isfile(os.path.join(cls_dir, f))]
    selected = random.sample(files, limit_per_class) if len(files) > limit_per_class else files
    for path in selected:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Aviso: arquivo inválido, pulando {path}")
            continue
        all_paths.append(path)
        all_labels.append(label)

# 3. Dataset customizado (sem preprocessing além de resize e normalização)
class CatDogDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths = paths
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, img_size)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        label = self.labels[idx]
        return torch.tensor(img), torch.tensor(label)

# 4. Cria dataset e faz split 80/20
dataset = CatDogDataset(all_paths, all_labels)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
datasets = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(datasets[0], batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(datasets[1], batch_size=batch_size)

# 5. Definição da CNN simples
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 6. Treinamento
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for imgs, labs in train_loader:
        imgs, labs = imgs.to(device), labs.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

# 7. Avaliação no conjunto de teste
y_true, y_pred = [], []
model.eval()
with torch.no_grad():
    for imgs, labs in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1).cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(labs.numpy().tolist())

print("=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=['cats','dogs']))
cm = confusion_matrix(y_true, y_pred)
print("=== Confusion Matrix ===")
print(cm)

# 8. Função para pré-processar e salvar novas imagens

def preprocess_and_save(input_paths, output_dir, size=(128,128), blur_ksize=(5,5)):
    """
    Recebe lista de caminhos de imagens, aplica redimensionamento, equalização de histograma
    e filtro gaussiano. Salva em output_dir mantendo subpastas 'cats' e 'dogs'.
    """
    os.makedirs(output_dir, exist_ok=True)
    for path in input_paths:
        # define classe pela pasta pai
        cls = os.path.basename(os.path.dirname(path))
        cls_dir = os.path.join(output_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        # lê em grayscale
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # redimensiona
        img = cv2.resize(img, size)
        # equaliza histograma
        img = cv2.equalizeHist(img)
        # aplica blur gaussiano
        img = cv2.GaussianBlur(img, blur_ksize, 0)
        # salva
        filename = os.path.basename(path)
        cv2.imwrite(os.path.join(cls_dir, filename), img)

# 9. Coleta aleatória de 6 cats e 6 dogs e salva em output_images
cats = [p for p, l in zip(all_paths, all_labels) if l==0]
dogs = [p for p, l in zip(all_paths, all_labels) if l==1]
sample_cats2 = random.sample(cats, 6)
sample_dogs2 = random.sample(dogs, 6)
preprocess_and_save(sample_cats2 + sample_dogs2, output_dir='output_images')

print("Imagens pré-processadas e salvas em 'output_images/'.")
