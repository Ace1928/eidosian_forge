import os
from functools import partial, reduce
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple, Type, Union
import transformers
from .. import PretrainedConfig, is_tf_available, is_torch_available
from ..utils import TF2_WEIGHTS_NAME, WEIGHTS_NAME, logging
from .config import OnnxConfig
@staticmethod
def get_model_class_for_feature(feature: str, framework: str='pt') -> Type:
    """
        Attempts to retrieve an AutoModel class from a feature name.

        Args:
            feature (`str`):
                The feature required.
            framework (`str`, *optional*, defaults to `"pt"`):
                The framework to use for the export.

        Returns:
            The AutoModel class corresponding to the feature.
        """
    task = FeaturesManager.feature_to_task(feature)
    FeaturesManager._validate_framework_choice(framework)
    if framework == 'pt':
        task_to_automodel = FeaturesManager._TASKS_TO_AUTOMODELS
    else:
        task_to_automodel = FeaturesManager._TASKS_TO_TF_AUTOMODELS
    if task not in task_to_automodel:
        raise KeyError(f'Unknown task: {feature}. Possible values are {list(FeaturesManager._TASKS_TO_AUTOMODELS.values())}')
    return task_to_automodel[task]