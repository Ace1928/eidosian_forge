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
def conversational(self, text: str, generated_responses: Optional[List[str]]=None, past_user_inputs: Optional[List[str]]=None, *, parameters: Optional[Dict[str, Any]]=None, model: Optional[str]=None) -> ConversationalOutput:
    """
        Generate conversational responses based on the given input text (i.e. chat with the API).

        Args:
            text (`str`):
                The last input from the user in the conversation.
            generated_responses (`List[str]`, *optional*):
                A list of strings corresponding to the earlier replies from the model. Defaults to None.
            past_user_inputs (`List[str]`, *optional*):
                A list of strings corresponding to the earlier replies from the user. Should be the same length as
                `generated_responses`. Defaults to None.
            parameters (`Dict[str, Any]`, *optional*):
                Additional parameters for the conversational task. Defaults to None. For more details about the available
                parameters, please refer to [this page](https://huggingface.co/docs/api-inference/detailed_parameters#conversational-task)
            model (`str`, *optional*):
                The model to use for the conversational task. Can be a model ID hosted on the Hugging Face Hub or a URL to
                a deployed Inference Endpoint. If not provided, the default recommended conversational model will be used.
                Defaults to None.

        Returns:
            `Dict`: The generated conversational output.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `HTTPError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> output = client.conversational("Hi, who are you?")
        >>> output
        {'generated_text': 'I am the one who knocks.', 'conversation': {'generated_responses': ['I am the one who knocks.'], 'past_user_inputs': ['Hi, who are you?']}, 'warnings': ['Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.']}
        >>> client.conversational(
        ...     "Wow, that's scary!",
        ...     generated_responses=output["conversation"]["generated_responses"],
        ...     past_user_inputs=output["conversation"]["past_user_inputs"],
        ... )
        ```
        """
    payload: Dict[str, Any] = {'inputs': {'text': text}}
    if generated_responses is not None:
        payload['inputs']['generated_responses'] = generated_responses
    if past_user_inputs is not None:
        payload['inputs']['past_user_inputs'] = past_user_inputs
    if parameters is not None:
        payload['parameters'] = parameters
    response = self.post(json=payload, model=model, task='conversational')
    return _bytes_to_dict(response)