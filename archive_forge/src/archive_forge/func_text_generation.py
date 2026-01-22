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
def text_generation(self, prompt: str, *, details: bool=False, stream: bool=False, model: Optional[str]=None, do_sample: bool=False, max_new_tokens: int=20, best_of: Optional[int]=None, repetition_penalty: Optional[float]=None, return_full_text: bool=False, seed: Optional[int]=None, stop_sequences: Optional[List[str]]=None, temperature: Optional[float]=None, top_k: Optional[int]=None, top_p: Optional[float]=None, truncate: Optional[int]=None, typical_p: Optional[float]=None, watermark: bool=False, decoder_input_details: bool=False) -> Union[str, TextGenerationResponse, Iterable[str], Iterable[TextGenerationStreamResponse]]:
    """
        Given a prompt, generate the following text.

        It is recommended to have Pydantic installed in order to get inputs validated. This is preferable as it allow
        early failures.

        API endpoint is supposed to run with the `text-generation-inference` backend (TGI). This backend is the
        go-to solution to run large language models at scale. However, for some smaller models (e.g. "gpt2") the
        default `transformers` + `api-inference` solution is still in use. Both approaches have very similar APIs, but
        not exactly the same. This method is compatible with both approaches but some parameters are only available for
        `text-generation-inference`. If some parameters are ignored, a warning message is triggered but the process
        continues correctly.

        To learn more about the TGI project, please refer to https://github.com/huggingface/text-generation-inference.

        Args:
            prompt (`str`):
                Input text.
            details (`bool`, *optional*):
                By default, text_generation returns a string. Pass `details=True` if you want a detailed output (tokens,
                probabilities, seed, finish reason, etc.). Only available for models running on with the
                `text-generation-inference` backend.
            stream (`bool`, *optional*):
                By default, text_generation returns the full generated text. Pass `stream=True` if you want a stream of
                tokens to be returned. Only available for models running on with the `text-generation-inference`
                backend.
            model (`str`, *optional*):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. This parameter overrides the model defined at the instance level. Defaults to None.
            do_sample (`bool`):
                Activate logits sampling
            max_new_tokens (`int`):
                Maximum number of generated tokens
            best_of (`int`):
                Generate best_of sequences and return the one if the highest token logprobs
            repetition_penalty (`float`):
                The parameter for repetition penalty. 1.0 means no penalty. See [this
                paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
            return_full_text (`bool`):
                Whether to prepend the prompt to the generated text
            seed (`int`):
                Random sampling seed
            stop_sequences (`List[str]`):
                Stop generating tokens if a member of `stop_sequences` is generated
            temperature (`float`):
                The value used to module the logits distribution.
            top_k (`int`):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (`float`):
                If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
                higher are kept for generation.
            truncate (`int`):
                Truncate inputs tokens to the given size
            typical_p (`float`):
                Typical Decoding mass
                See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information
            watermark (`bool`):
                Watermarking with [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)
            decoder_input_details (`bool`):
                Return the decoder input token logprobs and ids. You must set `details=True` as well for it to be taken
                into account. Defaults to `False`.

        Returns:
            `Union[str, TextGenerationResponse, Iterable[str], Iterable[TextGenerationStreamResponse]]`:
            Generated text returned from the server:
            - if `stream=False` and `details=False`, the generated text is returned as a `str` (default)
            - if `stream=True` and `details=False`, the generated text is returned token by token as a `Iterable[str]`
            - if `stream=False` and `details=True`, the generated text is returned with more details as a [`~huggingface_hub.inference._text_generation.TextGenerationResponse`]
            - if `details=True` and `stream=True`, the generated text is returned token by token as a iterable of [`~huggingface_hub.inference._text_generation.TextGenerationStreamResponse`]

        Raises:
            `ValidationError`:
                If input values are not valid. No HTTP call is made to the server.
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `HTTPError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()

        # Case 1: generate text
        >>> client.text_generation("The huggingface_hub library is ", max_new_tokens=12)
        '100% open source and built to be easy to use.'

        # Case 2: iterate over the generated tokens. Useful for large generation.
        >>> for token in client.text_generation("The huggingface_hub library is ", max_new_tokens=12, stream=True):
        ...     print(token)
        100
        %
        open
        source
        and
        built
        to
        be
        easy
        to
        use
        .

        # Case 3: get more details about the generation process.
        >>> client.text_generation("The huggingface_hub library is ", max_new_tokens=12, details=True)
        TextGenerationResponse(
            generated_text='100% open source and built to be easy to use.',
            details=Details(
                finish_reason=<FinishReason.Length: 'length'>,
                generated_tokens=12,
                seed=None,
                prefill=[
                    InputToken(id=487, text='The', logprob=None),
                    InputToken(id=53789, text=' hugging', logprob=-13.171875),
                    (...)
                    InputToken(id=204, text=' ', logprob=-7.0390625)
                ],
                tokens=[
                    Token(id=1425, text='100', logprob=-1.0175781, special=False),
                    Token(id=16, text='%', logprob=-0.0463562, special=False),
                    (...)
                    Token(id=25, text='.', logprob=-0.5703125, special=False)
                ],
                best_of_sequences=None
            )
        )

        # Case 4: iterate over the generated tokens with more details.
        # Last object is more complete, containing the full generated text and the finish reason.
        >>> for details in client.text_generation("The huggingface_hub library is ", max_new_tokens=12, details=True, stream=True):
        ...     print(details)
        ...
        TextGenerationStreamResponse(token=Token(id=1425, text='100', logprob=-1.0175781, special=False), generated_text=None, details=None)
        TextGenerationStreamResponse(token=Token(id=16, text='%', logprob=-0.0463562, special=False), generated_text=None, details=None)
        TextGenerationStreamResponse(token=Token(id=1314, text=' open', logprob=-1.3359375, special=False), generated_text=None, details=None)
        TextGenerationStreamResponse(token=Token(id=3178, text=' source', logprob=-0.28100586, special=False), generated_text=None, details=None)
        TextGenerationStreamResponse(token=Token(id=273, text=' and', logprob=-0.5961914, special=False), generated_text=None, details=None)
        TextGenerationStreamResponse(token=Token(id=3426, text=' built', logprob=-1.9423828, special=False), generated_text=None, details=None)
        TextGenerationStreamResponse(token=Token(id=271, text=' to', logprob=-1.4121094, special=False), generated_text=None, details=None)
        TextGenerationStreamResponse(token=Token(id=314, text=' be', logprob=-1.5224609, special=False), generated_text=None, details=None)
        TextGenerationStreamResponse(token=Token(id=1833, text=' easy', logprob=-2.1132812, special=False), generated_text=None, details=None)
        TextGenerationStreamResponse(token=Token(id=271, text=' to', logprob=-0.08520508, special=False), generated_text=None, details=None)
        TextGenerationStreamResponse(token=Token(id=745, text=' use', logprob=-0.39453125, special=False), generated_text=None, details=None)
        TextGenerationStreamResponse(token=Token(
            id=25,
            text='.',
            logprob=-0.5703125,
            special=False),
            generated_text='100% open source and built to be easy to use.',
            details=StreamDetails(finish_reason=<FinishReason.Length: 'length'>, generated_tokens=12, seed=None)
        )
        ```
        """
    if decoder_input_details and (not details):
        warnings.warn('`decoder_input_details=True` has been passed to the server but `details=False` is set meaning that the output from the server will be truncated.')
        decoder_input_details = False
    parameters = TextGenerationParameters(best_of=best_of, details=details, do_sample=do_sample, max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty, return_full_text=return_full_text, seed=seed, stop=stop_sequences if stop_sequences is not None else [], temperature=temperature, top_k=top_k, top_p=top_p, truncate=truncate, typical_p=typical_p, watermark=watermark, decoder_input_details=decoder_input_details)
    request = TextGenerationRequest(inputs=prompt, stream=stream, parameters=parameters)
    payload = asdict(request)
    if not _is_tgi_server(model):
        ignored_parameters = []
        for key in ('watermark', 'stop', 'details', 'decoder_input_details', 'best_of'):
            if payload['parameters'][key] is not None:
                ignored_parameters.append(key)
            del payload['parameters'][key]
        if len(ignored_parameters) > 0:
            warnings.warn(f'API endpoint/model for text-generation is not served via TGI. Ignoring parameters {ignored_parameters}.', UserWarning)
        if details:
            warnings.warn('API endpoint/model for text-generation is not served via TGI. Parameter `details=True` will be ignored meaning only the generated text will be returned.', UserWarning)
            details = False
        if stream:
            raise ValueError('API endpoint/model for text-generation is not served via TGI. Cannot return output as a stream. Please pass `stream=False` as input.')
    try:
        bytes_output = self.post(json=payload, model=model, task='text-generation', stream=stream)
    except HTTPError as e:
        if isinstance(e, BadRequestError) and 'The following `model_kwargs` are not used by the model' in str(e):
            _set_as_non_tgi(model)
            return self.text_generation(prompt=prompt, details=details, stream=stream, model=model, do_sample=do_sample, max_new_tokens=max_new_tokens, best_of=best_of, repetition_penalty=repetition_penalty, return_full_text=return_full_text, seed=seed, stop_sequences=stop_sequences, temperature=temperature, top_k=top_k, top_p=top_p, truncate=truncate, typical_p=typical_p, watermark=watermark, decoder_input_details=decoder_input_details)
        raise_text_generation_error(e)
    if stream:
        return _stream_text_generation_response(bytes_output, details)
    data = _bytes_to_dict(bytes_output)[0]
    return TextGenerationResponse(**data) if details else data['generated_text']