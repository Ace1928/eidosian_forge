import os
import warnings
from collections import defaultdict
from concurrent import futures
from typing import Any, Callable, Optional, Tuple
from warnings import warn
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import whoami
from ..models import DDPOStableDiffusionPipeline
from . import BaseTrainer, DDPOConfig
from .utils import PerPromptStatTracker
def create_model_card(self, path: str, model_name: Optional[str]='TRL DDPO Model') -> None:
    """Creates and saves a model card for a TRL model.

        Args:
            path (`str`): The path to save the model card to.
            model_name (`str`, *optional*): The name of the model, defaults to `TRL DDPO Model`.
        """
    try:
        user = whoami()['name']
    except Exception:
        warnings.warn('Cannot retrieve user information assuming you are running in offline mode.')
        return
    if not os.path.exists(path):
        os.makedirs(path)
    model_card_content = MODEL_CARD_TEMPLATE.format(model_name=model_name, model_id=f'{user}/{path}')
    with open(os.path.join(path, 'README.md'), 'w', encoding='utf-8') as f:
        f.write(model_card_content)