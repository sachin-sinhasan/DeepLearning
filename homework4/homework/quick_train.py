#!/usr/bin/env python3
"""
Quick training launcher for planner models.
Usage: python3 -m homework.quick_train [model_name]
"""

import sys
import subprocess
from pathlib import Path

from .train_configs import ALL_CONFIGS, print_training_commands


def main():
    if len(sys.argv) != 2 or sys.argv[1] in ['-h', '--help', 'help']:
        print("Usage: python3 -m homework.quick_train [model_name]")
        print("\nAvailable models:")
        for name in ALL_CONFIGS.keys():
            print(f"  {name}")
        print("\nFor detailed configs, run: python3 -m homework.train_configs")
        print("\nExamples:")
        print("  python3 -m homework.quick_train mlp")
        print("  python3 -m homework.quick_train transformer")
        print("  python3 -m homework.quick_train cnn")
        return
    
    model_name = sys.argv[1].lower()
    
    if model_name not in ALL_CONFIGS:
        print(f"Unknown model: {model_name}")
        print("Available models:", list(ALL_CONFIGS.keys()))
        return
    
    config = ALL_CONFIGS[model_name]
    print(f"Starting training for {model_name.upper()}...")
    print(f"Description: {config['description']}")
    print(f"Parameters: epochs={config['epochs']}, lr={config['lr']}, "
          f"batch_size={config['batch_size']}")
    print("-" * 50)
    
    # Build command
    cmd = ["python3", "-m", "homework.train_planner"]
    for key, value in config.items():
        if key != "description":
            cmd.extend([f"--{key}", str(value)])
    
    # Add log file
    log_file = f"logs/{model_name}_training.csv"
    cmd.extend(["--log_file", log_file])
    
    print(f"Running: {' '.join(cmd)}")
    print(f"Logs will be saved to: {log_file}")
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Start training
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")


if __name__ == "__main__":
    main()
