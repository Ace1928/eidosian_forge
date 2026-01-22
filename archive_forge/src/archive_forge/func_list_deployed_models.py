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
def list_deployed_models(self, frameworks: Union[None, str, Literal['all'], List[str]]=None) -> Dict[str, List[str]]:
    """
        List models currently deployed on the Inference API service.

        This helper checks deployed models framework by framework. By default, it will check the 4 main frameworks that
        are supported and account for 95% of the hosted models. However, if you want a complete list of models you can
        specify `frameworks="all"` as input. Alternatively, if you know before-hand which framework you are interested
        in, you can also restrict to search to this one (e.g. `frameworks="text-generation-inference"`). The more
        frameworks are checked, the more time it will take.

        <Tip>

        This endpoint is mostly useful for discoverability. If you already know which model you want to use and want to
        check its availability, you can directly use [`~InferenceClient.get_model_status`].

        </Tip>

        Args:
            frameworks (`Literal["all"]` or `List[str]` or `str`, *optional*):
                The frameworks to filter on. By default only a subset of the available frameworks are tested. If set to
                "all", all available frameworks will be tested. It is also possible to provide a single framework or a
                custom set of frameworks to check.

        Returns:
            `Dict[str, List[str]]`: A dictionary mapping task names to a sorted list of model IDs.

        Example:
        ```python
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()

        # Discover zero-shot-classification models currently deployed
        >>> models = client.list_deployed_models()
        >>> models["zero-shot-classification"]
        ['Narsil/deberta-large-mnli-zero-cls', 'facebook/bart-large-mnli', ...]

        # List from only 1 framework
        >>> client.list_deployed_models("text-generation-inference")
        {'text-generation': ['bigcode/starcoder', 'meta-llama/Llama-2-70b-chat-hf', ...], ...}
        ```
        """
    if frameworks is None:
        frameworks = MAIN_INFERENCE_API_FRAMEWORKS
    elif frameworks == 'all':
        frameworks = ALL_INFERENCE_API_FRAMEWORKS
    elif isinstance(frameworks, str):
        frameworks = [frameworks]
    frameworks = list(set(frameworks))
    models_by_task: Dict[str, List[str]] = {}

    def _unpack_response(framework: str, items: List[Dict]) -> None:
        for model in items:
            if framework == 'sentence-transformers':
                models_by_task.setdefault('feature-extraction', []).append(model['model_id'])
                models_by_task.setdefault('sentence-similarity', []).append(model['model_id'])
            else:
                models_by_task.setdefault(model['task'], []).append(model['model_id'])
    for framework in frameworks:
        response = get_session().get(f'{INFERENCE_ENDPOINT}/framework/{framework}', headers=self.headers)
        hf_raise_for_status(response)
        _unpack_response(framework, response.json())
    for task, models in models_by_task.items():
        models_by_task[task] = sorted(set(models), key=lambda x: x.lower())
    return models_by_task