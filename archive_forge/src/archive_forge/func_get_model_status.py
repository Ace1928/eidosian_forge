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
def get_model_status(self, model: Optional[str]=None) -> ModelStatus:
    """
        Get the status of a model hosted on the Inference API.

        <Tip>

        This endpoint is mostly useful when you already know which model you want to use and want to check its
        availability. If you want to discover already deployed models, you should rather use [`~InferenceClient.list_deployed_models`].

        </Tip>

        Args:
            model (`str`, *optional*):
                Identifier of the model for witch the status gonna be checked. If model is not provided,
                the model associated with this instance of [`InferenceClient`] will be used. Only InferenceAPI service can be checked so the
                identifier cannot be a URL.


        Returns:
            [`ModelStatus`]: An instance of ModelStatus dataclass, containing information,
                         about the state of the model: load, state, compute type and framework.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> client.get_model_status("bigcode/starcoder")
        ModelStatus(loaded=True, state='Loaded', compute_type='gpu', framework='text-generation-inference')
        ```
        """
    model = model or self.model
    if model is None:
        raise ValueError('Model id not provided.')
    if model.startswith('https://'):
        raise NotImplementedError('Model status is only available for Inference API endpoints.')
    url = f'{INFERENCE_ENDPOINT}/status/{model}'
    response = get_session().get(url, headers=self.headers)
    hf_raise_for_status(response)
    response_data = response.json()
    if 'error' in response_data:
        raise ValueError(response_data['error'])
    return ModelStatus(loaded=response_data['loaded'], state=response_data['state'], compute_type=response_data['compute_type'], framework=response_data['framework'])