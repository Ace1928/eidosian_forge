import os
from functools import partial, reduce
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple, Type, Union
import transformers
from .. import PretrainedConfig, is_tf_available, is_torch_available
from ..utils import TF2_WEIGHTS_NAME, WEIGHTS_NAME, logging
from .config import OnnxConfig
@staticmethod
def get_supported_features_for_model_type(model_type: str, model_name: Optional[str]=None) -> Dict[str, Callable[[PretrainedConfig], OnnxConfig]]:
    """
        Tries to retrieve the feature -> OnnxConfig constructor map from the model type.

        Args:
            model_type (`str`):
                The model type to retrieve the supported features for.
            model_name (`str`, *optional*):
                The name attribute of the model object, only used for the exception message.

        Returns:
            The dictionary mapping each feature to a corresponding OnnxConfig constructor.
        """
    model_type = model_type.lower()
    if model_type not in FeaturesManager._SUPPORTED_MODEL_TYPE:
        model_type_and_model_name = f'{model_type} ({model_name})' if model_name else model_type
        raise KeyError(f'{model_type_and_model_name} is not supported yet. Only {list(FeaturesManager._SUPPORTED_MODEL_TYPE.keys())} are supported. If you want to support {model_type} please propose a PR or open up an issue.')
    return FeaturesManager._SUPPORTED_MODEL_TYPE[model_type]