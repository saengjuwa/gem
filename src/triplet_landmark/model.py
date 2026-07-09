from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _conv_block(in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
    )


class LandmarkEmbeddingNet(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int = 256) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.features = nn.Sequential(
            _conv_block(3, 32, stride=2),
            _conv_block(32, 64, stride=2),
            _conv_block(64, 128, stride=2),
            _conv_block(128, 192, stride=2),
            _conv_block(192, 256, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.embedding = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(inplace=True),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(images)
        embedding = self.embedding(features)
        embedding = F.normalize(embedding, p=2, dim=1)
        logits = self.classifier(embedding)
        return embedding, logits


class GeMPool2d(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.tensor([p], dtype=torch.float32))
        self.eps = eps

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        p = self.p.clamp(min=self.eps)
        pooled = F.avg_pool2d(
            features.clamp(min=self.eps).pow(p),
            kernel_size=(features.size(-2), features.size(-1)),
        )
        return pooled.pow(1.0 / p).flatten(1)


class TimmEmbeddingNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        model_name: str,
        pretrained: bool,
        use_projection: bool = False,
        pooling: str = "avg",
        gem_p: float = 3.0,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("Install timm to use --model-name other than small_cnn.") from exc

        self.use_projection = use_projection
        self.pooling = pooling
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="" if pooling == "gem" else "avg",
        )
        self.pool = GeMPool2d(gem_p) if pooling == "gem" else nn.Identity()
        feature_dim = int(self.backbone.num_features)
        if use_projection:
            self.embedding_dim = embedding_dim
            self.embedding = nn.Sequential(
                nn.Linear(feature_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.SiLU(inplace=True),
            )
        else:
            self.embedding_dim = feature_dim
            self.embedding = nn.Identity()
        self.classifier = nn.Linear(self.embedding_dim, num_classes)

    def _to_channels_first(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            return features
        if features.shape[1] == self.backbone.num_features:
            return features
        if features.shape[-1] == self.backbone.num_features:
            return features.permute(0, 3, 1, 2).contiguous()
        return features

    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        if self.pooling == "avg":
            return self.backbone(images)

        features = self.backbone.forward_features(images)
        if features.ndim == 4:
            return self.pool(self._to_channels_first(features))
        if features.ndim == 3:
            token_features = features[:, 1:] if features.size(1) > 1 else features
            return token_features.mean(dim=1)
        return features

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._extract_features(images)
        embedding = self.embedding(features)
        embedding = F.normalize(embedding, p=2, dim=1)
        logits = self.classifier(embedding)
        return embedding, logits


def create_model(
    num_classes: int,
    embedding_dim: int = 256,
    model_name: str = "small_cnn",
    pretrained: bool = False,
    use_projection: bool = False,
    pooling: str = "avg",
    gem_p: float = 3.0,
) -> nn.Module:
    if model_name == "small_cnn":
        return LandmarkEmbeddingNet(num_classes=num_classes, embedding_dim=embedding_dim)
    return TimmEmbeddingNet(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        model_name=model_name,
        pretrained=pretrained,
        use_projection=use_projection,
        pooling=pooling,
        gem_p=gem_p,
    )
