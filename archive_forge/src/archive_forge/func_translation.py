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
def translation(self, text: str, *, model: Optional[str]=None, src_lang: Optional[str]=None, tgt_lang: Optional[str]=None) -> str:
    """
        Convert text from one language to another.

        Check out https://huggingface.co/tasks/translation for more information on how to choose the best model for
        your specific use case. Source and target languages usually depend on the model.
        However, it is possible to specify source and target languages for certain models. If you are working with one of these models,
        you can use `src_lang` and `tgt_lang` arguments to pass the relevant information.
        You can find this information in the model card.

        Args:
            text (`str`):
                A string to be translated.
            model (`str`, *optional*):
                The model to use for the translation task. Can be a model ID hosted on the Hugging Face Hub or a URL to
                a deployed Inference Endpoint. If not provided, the default recommended translation model will be used.
                Defaults to None.
            src_lang (`str`, *optional*):
                Source language of the translation task, i.e. input language. Cannot be passed without `tgt_lang`.
            tgt_lang (`str`, *optional*):
                Target language of the translation task, i.e. output language. Cannot be passed without `src_lang`.

        Returns:
            `str`: The generated translated text.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `HTTPError`:
                If the request fails with an HTTP error status code other than HTTP 503.
            `ValueError`:
                If only one of the `src_lang` and `tgt_lang` arguments are provided.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> client.translation("My name is Wolfgang and I live in Berlin")
        'Mein Name ist Wolfgang und ich lebe in Berlin.'
        >>> client.translation("My name is Wolfgang and I live in Berlin", model="Helsinki-NLP/opus-mt-en-fr")
        "Je m'appelle Wolfgang et je vis à Berlin."
        ```

        Specifying languages:
        ```py
        >>> client.translation("My name is Sarah Jessica Parker but you can call me Jessica", model="facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX", tgt_lang="fr_XX")
        "Mon nom est Sarah Jessica Parker mais vous pouvez m'appeler Jessica"
        ```
        """
    if src_lang is not None and tgt_lang is None:
        raise ValueError('You cannot specify `src_lang` without specifying `tgt_lang`.')
    if src_lang is None and tgt_lang is not None:
        raise ValueError('You cannot specify `tgt_lang` without specifying `src_lang`.')
    payload: Dict = {'inputs': text}
    if src_lang and tgt_lang:
        payload['parameters'] = {'src_lang': src_lang, 'tgt_lang': tgt_lang}
    response = self.post(json=payload, model=model, task='translation')
    return _bytes_to_dict(response)[0]['translation_text']