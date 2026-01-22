import os
from functools import partial, reduce
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple, Type, Union
import transformers
from .. import PretrainedConfig, is_tf_available, is_torch_available
from ..utils import TF2_WEIGHTS_NAME, WEIGHTS_NAME, logging
from .config import OnnxConfig
def supported_features_mapping(*supported_features: str, onnx_config_cls: str=None) -> Dict[str, Callable[[PretrainedConfig], OnnxConfig]]:
    """
    Generate the mapping between supported the features and their corresponding OnnxConfig for a given model.

    Args:
        *supported_features: The names of the supported features.
        onnx_config_cls: The OnnxConfig full name corresponding to the model.

    Returns:
        The dictionary mapping a feature to an OnnxConfig constructor.
    """
    if onnx_config_cls is None:
        raise ValueError('A OnnxConfig class must be provided')
    config_cls = transformers
    for attr_name in onnx_config_cls.split('.'):
        config_cls = getattr(config_cls, attr_name)
    mapping = {}
    for feature in supported_features:
        if '-with-past' in feature:
            task = feature.replace('-with-past', '')
            mapping[feature] = partial(config_cls.with_past, task=task)
        else:
            mapping[feature] = partial(config_cls.from_model_config, task=feature)
    return mapping