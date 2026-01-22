import os
from functools import partial, reduce
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple, Type, Union
import transformers
from .. import PretrainedConfig, is_tf_available, is_torch_available
from ..utils import TF2_WEIGHTS_NAME, WEIGHTS_NAME, logging
from .config import OnnxConfig
@staticmethod
def get_model_from_feature(feature: str, model: str, framework: str=None, cache_dir: str=None) -> Union['PreTrainedModel', 'TFPreTrainedModel']:
    """
        Attempts to retrieve a model from a model's name and the feature to be enabled.

        Args:
            feature (`str`):
                The feature required.
            model (`str`):
                The name of the model to export.
            framework (`str`, *optional*, defaults to `None`):
                The framework to use for the export. See `FeaturesManager.determine_framework` for the priority should
                none be provided.

        Returns:
            The instance of the model.

        """
    framework = FeaturesManager.determine_framework(model, framework)
    model_class = FeaturesManager.get_model_class_for_feature(feature, framework)
    try:
        model = model_class.from_pretrained(model, cache_dir=cache_dir)
    except OSError:
        if framework == 'pt':
            logger.info('Loading TensorFlow model in PyTorch before exporting to ONNX.')
            model = model_class.from_pretrained(model, from_tf=True, cache_dir=cache_dir)
        else:
            logger.info('Loading PyTorch model in TensorFlow before exporting to ONNX.')
            model = model_class.from_pretrained(model, from_pt=True, cache_dir=cache_dir)
    return model