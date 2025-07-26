import torch
from homework.models import Classifier
from datasets.classification_dataset import load_data
from homework.metrics import AccuracyMetric
from homework.models import save_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_loader = load_data(
    "classification_data",
    transform_pipeline="aug",
    batch_size=128,
    shuffle=True,
)
val_loader = load_data(
    "classification_data",
    transform_pipeline="default",
    batch_size=128,
    shuffle=False,
)

# Model, optimizer, loss
model = Classifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()
metric = AccuracyMetric()

# Training loop
for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model.predict(images)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"Epoch {epoch+1}: Val Accuracy = {correct/total:.4f}")

# Save model
save_model(model, "classifier.pt")