import torch
from homework.models import Detector
from datasets.classification_dataset import load_data
from homework.metrics import ConfusionMatrix, MAEMetric
from homework.models import save_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_loader = load_data(
    "drive_data",
    batch_size=32,
    shuffle=True,
)
val_loader = load_data(
    "drive_data",
    batch_size=32,
    shuffle=False,
)

# Model, optimizer, loss
model = Detector().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
seg_loss = torch.nn.CrossEntropyLoss()
depth_loss = torch.nn.L1Loss()

# Training loop
for epoch in range(10):
    model.train()
    for batch in train_loader:
        images = batch["image"].to(device)
        seg_labels = batch["track"].to(device)
        depth_labels = batch["depth"].to(device)
        optimizer.zero_grad()
        logits, depth = model(images)
        loss1 = seg_loss(logits, seg_labels)
        loss2 = depth_loss(depth, depth_labels)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
    # Validation
    model.eval()
    cm = ConfusionMatrix(num_classes=3)
    mae_metric = MAEMetric()
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            seg_labels = batch["track"].to(device)
            depth_labels = batch["depth"].to(device)
            logits, depth = model(images)
            preds = logits.argmax(dim=1)
            cm.update(preds, seg_labels)
            mae_metric.update(depth, depth_labels, seg_labels)
    print(f"Epoch {epoch+1}: mIoU={cm.mean_iou():.4f}, Depth MAE={mae_metric.mae():.4f}")

# Save model
save_model(model, "detector.pt")