"""Red y conversión de los arrays preparados a su entrada."""

from torch import nn
from torchvision.models import DenseNet121_Weights, EfficientNet_B0_Weights, densenet121, efficientnet_b0


def build_model(architecture, pretrained):
    """Devuelve la red con una salida (logit de la clase positiva)."""
    if architecture == 'efficientnet_b0':
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        return model
    if architecture != 'densenet121':
        raise ValueError(f'Arquitectura no soportada: {architecture}')
    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    return model


def to_input(batch, mean, std):
    """Convierte N,H,W en [0, 1] a N,3,H,W con la media y el desvío de train."""
    return ((batch - mean) / std).unsqueeze(1).repeat(1, 3, 1, 1)
