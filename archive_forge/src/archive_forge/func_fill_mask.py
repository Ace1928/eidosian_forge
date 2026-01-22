import base64
import logging
import time
import warnings
from dataclasses import asdict
from typing import (
from requests import HTTPError
from requests.structures import CaseInsensitiveDict
from huggingface_hub.constants import ALL_INFERENCE_API_FRAMEWORKS, INFERENCE_ENDPOINT, MAIN_INFERENCE_API_FRAMEWORKS
from huggingface_hub.inference._common import (
from huggingface_hub.inference._text_generation import (
from huggingface_hub.inference._types import (
from huggingface_hub.utils import (
def fill_mask(self, text: str, *, model: Optional[str]=None) -> List[FillMaskOutput]:
    """
        Fill in a hole with a missing word (token to be precise).

        Args:
            text (`str`):
                a string to be filled from, must contain the [MASK] token (check model card for exact name of the mask).
            model (`str`, *optional*):
                The model to use for the fill mask task. Can be a model ID hosted on the Hugging Face Hub or a URL to
                a deployed Inference Endpoint. If not provided, the default recommended fill mask model will be used.
                Defaults to None.

        Returns:
            `List[Dict]`: a list of fill mask output dictionaries containing the predicted label, associated
            probability, token reference, and completed text.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `HTTPError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> client.fill_mask("The goal of life is <mask>.")
        [{'score': 0.06897063553333282,
        'token': 11098,
        'token_str': ' happiness',
        'sequence': 'The goal of life is happiness.'},
        {'score': 0.06554922461509705,
        'token': 45075,
        'token_str': ' immortality',
        'sequence': 'The goal of life is immortality.'}]
        ```
        """
    response = self.post(json={'inputs': text}, model=model, task='fill-mask')
    return _bytes_to_list(response)