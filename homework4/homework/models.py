from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        
        # Input: (B, n_track, 2) for both left and right tracks
        # Total input features: n_track * 2 * 2 = 40
        input_features = n_track * 2 * 2
        
        # MLP architecture
        self.mlp = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_waypoints * 2)
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        batch_size = track_left.shape[0]
        
        # Flatten the track points
        track_left_flat = track_left.view(batch_size, -1)  # (B, n_track * 2)
        track_right_flat = track_right.view(batch_size, -1)  # (B, n_track * 2)
        
        # Concatenate left and right tracks
        combined_input = torch.cat([track_left_flat, track_right_flat], dim=1)  # (B, n_track * 2 * 2)
        
        # Pass through MLP
        output = self.mlp(combined_input)  # (B, n_waypoints * 2)
        
        # Reshape to (B, n_waypoints, 2)
        waypoints = output.view(batch_size, self.n_waypoints, 2)
        
        return waypoints


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        # Learnable query embeddings for waypoints
        self.query_embed = nn.Embedding(n_waypoints, d_model)
        
        # Input projection for track points with larger capacity
        self.input_projection = nn.Sequential(
            nn.Linear(2, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        # Multiple attention layers for better representation
        self.attention1 = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            batch_first=True,
            dropout=0.1
        )
        
        self.attention2 = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            batch_first=True,
            dropout=0.1
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Output projection with more capacity
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2)
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        batch_size = track_left.shape[0]
        
        # Concatenate left and right tracks
        track_points = torch.cat([track_left, track_right], dim=1)  # (B, n_track*2, 2)
        
        # Project track points to d_model dimensions
        track_features = self.input_projection(track_points)  # (B, n_track*2, d_model)
        
        # Get query embeddings for waypoints
        query_indices = torch.arange(self.n_waypoints, device=track_features.device)
        waypoint_queries = self.query_embed(query_indices)  # (n_waypoints, d_model)
        waypoint_queries = waypoint_queries.unsqueeze(0).expand(batch_size, -1, -1)  # (B, n_waypoints, d_model)
        
        # First attention layer with residual connection
        waypoint_features1, _ = self.attention1(
            query=waypoint_queries,  # (B, n_waypoints, d_model)
            key=track_features,      # (B, n_track*2, d_model)
            value=track_features     # (B, n_track*2, d_model)
        )
        waypoint_features1 = self.norm1(waypoint_queries + waypoint_features1)
        
        # Second attention layer
        waypoint_features2, _ = self.attention2(
            query=waypoint_features1,  # (B, n_waypoints, d_model)
            key=track_features,        # (B, n_track*2, d_model)
            value=track_features       # (B, n_track*2, d_model)
        )
        waypoint_features2 = self.norm2(waypoint_features1 + waypoint_features2)
        
        # Feed-forward network with residual connection
        waypoint_features3 = self.ffn(waypoint_features2)
        waypoint_features3 = self.norm3(waypoint_features2 + waypoint_features3)
        
        # Project to output coordinates
        waypoints = self.output_projection(waypoint_features3)  # (B, n_waypoints, 2)
        
        return waypoints


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)
        
        # Optimized CNN backbone - lighter but still effective
        self.features = nn.Sequential(
            # Input: (B, 3, 96, 128)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (B, 32, 48, 64)
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (B, 64, 24, 32)
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (B, 128, 12, 16)
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (B, 256, 6, 8)
        )
        
        # Streamlined classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # -> (B, 256, 1, 1)
            nn.Flatten(),  # -> (B, 256)
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_waypoints * 2)  # -> (B, n_waypoints * 2)
        )

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Extract features
        x = self.features(x)  # -> (B, 256, 6, 8)
        
        # Classify to waypoints
        x = self.classifier(x)  # -> (B, n_waypoints * 2)
        
        # Reshape to (B, n_waypoints, 2)
        waypoints = x.view(x.size(0), self.n_waypoints, 2)
        
        return waypoints


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
