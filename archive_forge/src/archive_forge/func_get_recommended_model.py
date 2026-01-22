import asyncio
import base64
import logging
import time
import warnings
from dataclasses import asdict
from typing import (
from requests.structures import CaseInsensitiveDict
from huggingface_hub.constants import ALL_INFERENCE_API_FRAMEWORKS, INFERENCE_ENDPOINT, MAIN_INFERENCE_API_FRAMEWORKS
from huggingface_hub.inference._common import (
from huggingface_hub.inference._text_generation import (
from huggingface_hub.inference._types import (
from huggingface_hub.utils import (
from .._common import _async_yield_from, _import_aiohttp
@staticmethod
def get_recommended_model(task: str) -> str:
    """
        Get the model Hugging Face recommends for the input task.

        Args:
            task (`str`):
                The Hugging Face task to get which model Hugging Face recommends.
                All available tasks can be found [here](https://huggingface.co/tasks).

        Returns:
            `str`: Name of the model recommended for the input task.

        Raises:
            `ValueError`: If Hugging Face has no recommendation for the input task.
        """
    model = _fetch_recommended_models().get(task)
    if model is None:
        raise ValueError(f'Task {task} has no recommended model. Please specify a model explicitly. Visit https://huggingface.co/tasks for more info.')
    return model