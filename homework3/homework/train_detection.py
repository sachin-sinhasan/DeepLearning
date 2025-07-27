import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from pathlib import Path

# Import our custom modules
from models import Detector, save_model
from metrics import DetectionMetric
from datasets.road_dataset import load_data


def train_detection():
    """
    Training script for road detection task (semantic segmentation + depth estimation)
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters
    num_epochs = 25  # More epochs for better performance
    batch_size = 8   # Smaller batch size for better gradient updates
    learning_rate = 0.0005  # Lower learning rate for better convergence
    num_classes = 3  # background, left boundary, right boundary
    
    # Data paths
    train_data_path = "../drive_data/train"
    val_data_path = "../drive_data/val"
    
    # Create model
    model = Detector(in_channels=3, num_classes=num_classes).to(device)
    
    # Create loss functions
    # Segmentation loss (CrossEntropyLoss for semantic segmentation)
    seg_criterion = nn.CrossEntropyLoss()
    
    # Depth loss (L1Loss for depth regression)
    depth_criterion = nn.L1Loss()
    
    # Combined loss weight - focus more on segmentation for better IoU
    depth_weight = 0.02  # Even less weight on depth to focus on segmentation
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create metrics
    train_metric = DetectionMetric(num_classes=num_classes)
    val_metric = DetectionMetric(num_classes=num_classes)
    
    # Load data
    train_loader = load_data(
        dataset_path=train_data_path,
        transform_pipeline="aug",  # Use augmentation for better performance
        return_dataloader=True,
        num_workers=2,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = load_data(
        dataset_path=val_data_path,
        transform_pipeline="default",
        return_dataloader=True,
        num_workers=2,
        batch_size=batch_size,
        shuffle=False
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Training loop
    best_val_iou = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_metric.reset()
        train_seg_loss = 0.0
        train_depth_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            # Extract data from batch
            images = batch["image"].to(device)  # (B, 3, H, W)
            depth_labels = batch["depth"].to(device)  # (B, H, W)
            track_labels = batch["track"].to(device)  # (B, H, W) - semantic segmentation labels
            
            # Forward pass
            optimizer.zero_grad()
            seg_logits, depth_preds = model(images)  # (B, num_classes, H, W), (B, H, W)
            
            # Compute losses
            seg_loss = seg_criterion(seg_logits, track_labels)
            depth_loss = depth_criterion(depth_preds, depth_labels)
            
            # Combined loss
            total_loss = seg_loss + depth_weight * depth_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Update metrics
            train_seg_loss += seg_loss.item()
            train_depth_loss += depth_loss.item()
            
            # Get predictions for metrics
            seg_preds = seg_logits.argmax(dim=1)  # (B, H, W)
            train_metric.add(seg_preds, track_labels, depth_preds, depth_labels)
            
            # Print progress
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Seg Loss: {seg_loss.item():.4f}, Depth Loss: {depth_loss.item():.4f}")
        
        # Validation phase
        model.eval()
        val_metric.reset()
        val_seg_loss = 0.0
        val_depth_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                # Extract data from batch
                images = batch["image"].to(device)
                depth_labels = batch["depth"].to(device)
                track_labels = batch["track"].to(device)
                
                # Forward pass
                seg_logits, depth_preds = model(images)
                
                # Compute losses
                seg_loss = seg_criterion(seg_logits, track_labels)
                depth_loss = depth_criterion(depth_preds, depth_labels)
                
                val_seg_loss += seg_loss.item()
                val_depth_loss += depth_loss.item()
                
                # Get predictions for metrics
                seg_preds = seg_logits.argmax(dim=1)
                val_metric.add(seg_preds, track_labels, depth_preds, depth_labels)
        
        # Compute metrics
        train_metrics = train_metric.compute()
        val_metrics = val_metric.compute()
        
        avg_train_seg_loss = train_seg_loss / len(train_loader)
        avg_train_depth_loss = train_depth_loss / len(train_loader)
        avg_val_seg_loss = val_seg_loss / len(val_loader)
        avg_val_depth_loss = val_depth_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train - Seg Loss: {avg_train_seg_loss:.4f}, Depth Loss: {avg_train_depth_loss:.4f}")
        print(f"  Train - IoU: {train_metrics['iou']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"  Train - Depth Error: {train_metrics['abs_depth_error']:.4f}")
        print(f"  Val - Seg Loss: {avg_val_seg_loss:.4f}, Depth Loss: {avg_val_depth_loss:.4f}")
        print(f"  Val - IoU: {val_metrics['iou']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Val - Depth Error: {val_metrics['abs_depth_error']:.4f}")
        print("-" * 50)
        
        # Save best model based on IoU
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            save_model(model)
            print(f"New best model saved with validation IoU: {best_val_iou:.4f}")
    
    print(f"Training completed! Best validation IoU: {best_val_iou:.4f}")


if __name__ == "__main__":
    train_detection()
