import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from pathlib import Path

# Import our custom modules
from models import Classifier, save_model
from metrics import AccuracyMetric
from datasets.classification_dataset import load_data


def train_classification():
    """
    Training script for classification task
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters
    num_epochs = 5  # Reduced for quick testing
    batch_size = 32
    learning_rate = 0.001
    num_classes = 6
    
    # Data paths
    train_data_path = "../classification_data/train"
    val_data_path = "../classification_data/val"
    
    # Create model
    model = Classifier(in_channels=3, num_classes=num_classes).to(device)
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create metrics
    train_metric = AccuracyMetric()
    val_metric = AccuracyMetric()
    
    # Load data
    train_loader = load_data(
        dataset_path=train_data_path,
        transform_pipeline="aug",  # Use augmentation for training
        return_dataloader=True,
        num_workers=2,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = load_data(
        dataset_path=val_data_path,
        transform_pipeline="default",  # No augmentation for validation
        return_dataloader=True,
        num_workers=2,
        batch_size=batch_size,
        shuffle=False
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Training loop
    best_val_accuracy = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_metric.reset()
        train_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            predictions = model.predict(images)
            train_metric.add(predictions, labels)
            
            # Print progress
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}")
        
        # Validation phase
        model.eval()
        val_metric.reset()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                predictions = model.predict(images)
                val_metric.add(predictions, labels)
        
        # Compute metrics
        train_metrics = train_metric.compute()
        val_metrics = val_metric.compute()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_metrics['accuracy']:.4f}")
        print("-" * 50)
        
        # Save best model
        if val_metrics['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_metrics['accuracy']
            save_model(model)
            print(f"New best model saved with validation accuracy: {best_val_accuracy:.4f}")
    
    print(f"Training completed! Best validation accuracy: {best_val_accuracy:.4f}")


if __name__ == "__main__":
    train_classification()
