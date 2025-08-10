"""
Usage:
    python3 -m homework.train_planner --model mlp_planner --epochs 100 --lr 0.001
    python3 -m homework.train_planner --model transformer_planner --epochs 100 --lr 0.0001
    python3 -m homework.train_planner --model cnn_planner --epochs 100 --lr 0.001
"""

import argparse
import os
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import road_dataset
from .models import load_model, save_model, MLPPlanner, TransformerPlanner
from .metrics import PlannerMetric


class TrainingLogger:
    """Simple training logger for tracking metrics"""
    
    def __init__(self, log_file=None):
        self.log_file = log_file
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'longitudinal_error': [],
            'lateral_error': [],
            'learning_rate': []
        }
        
    def log(self, epoch, train_loss, val_loss, val_metrics, lr):
        """Log training metrics"""
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['longitudinal_error'].append(val_metrics['longitudinal_error'])
        self.history['lateral_error'].append(val_metrics['lateral_error'])
        self.history['learning_rate'].append(lr)
        
        # Print to console
        print(f"Epoch {epoch:3d} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"Lon Error: {val_metrics['longitudinal_error']:.6f} | "
              f"Lat Error: {val_metrics['lateral_error']:.6f} | "
              f"LR: {lr:.6f}")
        
        # Write to file if specified
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},"
                       f"{val_metrics['longitudinal_error']:.6f},"
                       f"{val_metrics['lateral_error']:.6f},{lr:.6f}\n")


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress_bar:
        # Move data to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Get inputs and targets
        if isinstance(model, (MLPPlanner, TransformerPlanner)):
            # For MLP and Transformer planners
            track_left = batch["track_left"]
            track_right = batch["track_right"]
            waypoints = batch["waypoints"]
            waypoints_mask = batch["waypoints_mask"]
            
            # Forward pass
            pred = model(track_left, track_right)
        else:
            # For CNN planner
            image = batch["image"]
            waypoints = batch["waypoints"]
            waypoints_mask = batch["waypoints_mask"]
            
            # Forward pass
            pred = model(image)
        
        # Apply mask to predictions and targets
        pred_masked = pred * waypoints_mask[..., None]
        waypoints_masked = waypoints * waypoints_mask[..., None]
        
        # Compute loss
        loss = criterion(pred_masked, waypoints_masked)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


def evaluate(model, dataloader, device):
    """Evaluate the model"""
    model.eval()
    metric = PlannerMetric()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            # Move data to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Get inputs and targets
            if isinstance(model, (MLPPlanner, TransformerPlanner)):
                # For MLP and Transformer planners
                track_left = batch["track_left"]
                track_right = batch["track_right"]
                waypoints = batch["waypoints"]
                waypoints_mask = batch["waypoints_mask"]
                
                # Forward pass
                pred = model(track_left, track_right)
            else:
                # For CNN planner
                image = batch["image"]
                waypoints = batch["waypoints"]
                waypoints_mask = batch["waypoints_mask"]
                
                # Forward pass
                pred = model(image)
            
            # Add to metrics
            metric.add(pred, waypoints, waypoints_mask)
    
    return metric.compute()


def main():
    parser = argparse.ArgumentParser(description="Train planner models")
    parser.add_argument("--model", type=str, required=True, 
                       choices=["mlp_planner", "transformer_planner", "cnn_planner"],
                       help="Model to train")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of data workers")
    parser.add_argument("--save_every", type=int, default=10, help="Save model every N epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--min_delta", type=float, default=1e-4, help="Minimum improvement for early stopping")
    parser.add_argument("--log_file", type=str, help="CSV file to log training metrics")
    
    args = parser.parse_args()
    
    # Create log file if specified
    if args.log_file:
        log_dir = Path(args.log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        # Write header
        with open(args.log_file, 'w') as f:
            f.write("epoch,train_loss,val_loss,longitudinal_error,lateral_error,learning_rate\n")
    
    # Set device - prefer CPU for transformer to avoid MPS issues
    if args.model == "transformer_planner":
        device = torch.device("cpu")
        print("Using device: cpu (forced for transformer to avoid MPS issues)")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else 
                             "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model, with_weights=False)
    model.to(device)
    print(f"Loaded {args.model}")
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Set transform pipeline based on model
    if args.model in ["mlp_planner", "transformer_planner"]:
        transform_pipeline = "state_only"
    else:  # cnn_planner
        transform_pipeline = "default"
    
    print(f"Using transform pipeline: {transform_pipeline}")
    
    # Load training data
    print("Loading training data...")
    train_data = road_dataset.load_data(
        "drive_data/train",
        transform_pipeline=transform_pipeline,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True
    )
    
    # Load validation data
    print("Loading validation data...")
    val_data = road_dataset.load_data(
        "drive_data/val",
        transform_pipeline=transform_pipeline,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )
    
    print(f"Training batches: {len(train_data)}")
    print(f"Validation batches: {len(val_data)}")
    
    # Loss function and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training logger
    logger = TrainingLogger(args.log_file)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 80)
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_data, criterion, optimizer, device)
        
        # Evaluate
        val_metrics = evaluate(model, val_data, device)
        val_loss = val_metrics["l1_error"]
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        logger.log(epoch + 1, train_loss, val_loss, val_metrics, current_lr)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss - args.min_delta:
            best_val_loss = val_loss
            save_model(model)
            print(f"  💾 Saved new best model with val loss: {val_loss:.6f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Save periodically
        if (epoch + 1) % args.save_every == 0:
            save_path = save_model(model)
            print(f"  📁 Saved model checkpoint: {save_path}")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n🛑 Early stopping triggered after {epoch + 1} epochs")
            break
        
        # Print epoch timing
        epoch_time = time.time() - epoch_start_time
        print(f"  ⏱️  Epoch time: {epoch_time:.1f}s")
    
    # Training completed
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"🎉 Training completed in {total_time/60:.1f} minutes!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    # Final evaluation
    print("\nRunning final evaluation...")
    final_metrics = evaluate(model, val_data, device)
    print(f"Final Longitudinal Error: {final_metrics['longitudinal_error']:.6f}")
    print(f"Final Lateral Error: {final_metrics['lateral_error']:.6f}")
    
    # Check if performance targets are met
    if args.model in ["mlp_planner", "transformer_planner"]:
        lon_target = 0.2
        lat_target = 0.6
    else:  # cnn_planner
        lon_target = 0.3
        lat_target = 0.45
    
    print(f"\nPerformance Targets:")
    print(f"  Longitudinal Error: {'✅' if final_metrics['longitudinal_error'] < lon_target else '❌'} "
          f"< {lon_target} (got {final_metrics['longitudinal_error']:.6f})")
    print(f"  Lateral Error: {'✅' if final_metrics['lateral_error'] < lat_target else '❌'} "
          f"< {lat_target} (got {final_metrics['lateral_error']:.6f})")


if __name__ == "__main__":
    main()