"""
Training configurations for different planner models.
These are recommended starting points that can be tuned based on results.
"""

# MLP Planner training configuration
MLP_CONFIG = {
    "model": "mlp_planner",
    "epochs": 150,
    "lr": 0.001,
    "batch_size": 64,
    "patience": 25,
    "min_delta": 1e-4,
    "description": "MLP Planner - Simple feedforward network for track-based planning"
}

# Transformer Planner training configuration  
TRANSFORMER_CONFIG = {
    "model": "transformer_planner",
    "epochs": 200,
    "lr": 0.0005,  # Lower LR for transformer
    "batch_size": 32,  # Smaller batch size for memory
    "patience": 30,
    "min_delta": 1e-4,
    "description": "Transformer Planner - Attention-based model for track-based planning"
}

# CNN Planner training configuration
CNN_CONFIG = {
    "model": "cnn_planner", 
    "epochs": 100,
    "lr": 0.001,
    "batch_size": 16,  # Smaller batch size for image processing
    "patience": 20,
    "min_delta": 1e-4,
    "description": "CNN Planner - Convolutional network for image-based planning"
}

# All configurations
ALL_CONFIGS = {
    "mlp": MLP_CONFIG,
    "transformer": TRANSFORMER_CONFIG, 
    "cnn": CNN_CONFIG
}

def get_training_command(config_name):
    """Generate training command for a given configuration"""
    if config_name not in ALL_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(ALL_CONFIGS.keys())}")
    
    config = ALL_CONFIGS[config_name]
    cmd = f"python3 -m homework.train_planner"
    
    for key, value in config.items():
        if key != "description":
            cmd += f" --{key} {value}"
    
    return cmd

def print_training_commands():
    """Print all available training commands"""
    print("Available training configurations:")
    print("=" * 50)
    
    for name, config in ALL_CONFIGS.items():
        print(f"\n{name.upper()}: {config['description']}")
        print(f"Command: {get_training_command(name)}")
        print(f"Parameters: epochs={config['epochs']}, lr={config['lr']}, "
              f"batch_size={config['batch_size']}, patience={config['patience']}")

if __name__ == "__main__":
    print_training_commands()
