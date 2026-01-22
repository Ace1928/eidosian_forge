from os.path import join
from typing import Dict, Optional, Tuple, Union
import onnxruntime as ort
from numpy import ndarray
@property
def text_features_dim(self) -> int:
    """Dimensionality of the text encoder features."""
    return self._text_encoder_dim