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
def zero_shot_classification(self, text: str, labels: List[str], *, multi_label: bool=False, model: Optional[str]=None) -> List[ClassificationOutput]:
    """
        Provide as input a text and a set of candidate labels to classify the input text.

        Args:
            text (`str`):
                The input text to classify.
            labels (`List[str]`):
                List of string possible labels. There must be at least 2 labels.
            multi_label (`bool`):
                Boolean that is set to True if classes can overlap.
            model (`str`, *optional*):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. This parameter overrides the model defined at the instance level. Defaults to None.

        Returns:
            `List[Dict]`: List of classification outputs containing the predicted labels and their confidence.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `HTTPError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> text = (
        ...     "A new model offers an explanation for how the Galilean satellites formed around the solar system's"
        ...     "largest world. Konstantin Batygin did not set out to solve one of the solar system's most puzzling"
        ...     " mysteries when he went for a run up a hill in Nice, France."
        ... )
        >>> labels = ["space & cosmos", "scientific discovery", "microbiology", "robots", "archeology"]
        >>> client.zero_shot_classification(text, labels)
        [
            {"label": "scientific discovery", "score": 0.7961668968200684},
            {"label": "space & cosmos", "score": 0.18570658564567566},
            {"label": "microbiology", "score": 0.00730885099619627},
            {"label": "archeology", "score": 0.006258360575884581},
            {"label": "robots", "score": 0.004559356719255447},
        ]
        >>> client.zero_shot_classification(text, labels, multi_label=True)
        [
            {"label": "scientific discovery", "score": 0.9829297661781311},
            {"label": "space & cosmos", "score": 0.755190908908844},
            {"label": "microbiology", "score": 0.0005462635890580714},
            {"label": "archeology", "score": 0.00047131875180639327},
            {"label": "robots", "score": 0.00030448526376858354},
        ]
        ```
        """
    if len(labels) < 2:
        raise ValueError('You must specify at least 2 classes to compare.')
    response = self.post(json={'inputs': text, 'parameters': {'candidate_labels': ','.join(labels), 'multi_label': multi_label}}, model=model, task='zero-shot-classification')
    output = _bytes_to_dict(response)
    return [{'label': label, 'score': score} for label, score in zip(output['labels'], output['scores'])]