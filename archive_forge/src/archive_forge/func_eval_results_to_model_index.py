import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from huggingface_hub.utils import logging, yaml_dump
def eval_results_to_model_index(model_name: str, eval_results: List[EvalResult]) -> List[Dict[str, Any]]:
    """Takes in given model name and list of `huggingface_hub.EvalResult` and returns a
    valid model-index that will be compatible with the format expected by the
    Hugging Face Hub.

    Args:
        model_name (`str`):
            Name of the model (ex. "my-cool-model"). This is used as the identifier
            for the model on leaderboards like PapersWithCode.
        eval_results (`List[EvalResult]`):
            List of `huggingface_hub.EvalResult` objects containing the metrics to be
            reported in the model-index.

    Returns:
        model_index (`List[Dict[str, Any]]`): The eval_results converted to a model-index.

    Example:
        ```python
        >>> from huggingface_hub.repocard_data import eval_results_to_model_index, EvalResult
        >>> # Define minimal eval_results
        >>> eval_results = [
        ...     EvalResult(
        ...         task_type="image-classification",  # Required
        ...         dataset_type="beans",  # Required
        ...         dataset_name="Beans",  # Required
        ...         metric_type="accuracy",  # Required
        ...         metric_value=0.9,  # Required
        ...     )
        ... ]
        >>> eval_results_to_model_index("my-cool-model", eval_results)
        [{'name': 'my-cool-model', 'results': [{'task': {'type': 'image-classification'}, 'dataset': {'name': 'Beans', 'type': 'beans'}, 'metrics': [{'type': 'accuracy', 'value': 0.9}]}]}]

        ```
    """
    task_and_ds_types_map: Dict[Any, List[EvalResult]] = defaultdict(list)
    for eval_result in eval_results:
        task_and_ds_types_map[eval_result.unique_identifier].append(eval_result)
    model_index_data = []
    for results in task_and_ds_types_map.values():
        sample_result = results[0]
        data = {'task': {'type': sample_result.task_type, 'name': sample_result.task_name}, 'dataset': {'name': sample_result.dataset_name, 'type': sample_result.dataset_type, 'config': sample_result.dataset_config, 'split': sample_result.dataset_split, 'revision': sample_result.dataset_revision, 'args': sample_result.dataset_args}, 'metrics': [{'type': result.metric_type, 'value': result.metric_value, 'name': result.metric_name, 'config': result.metric_config, 'args': result.metric_args, 'verified': result.verified, 'verifyToken': result.verify_token} for result in results]}
        if sample_result.source_url is not None:
            source = {'url': sample_result.source_url}
            if sample_result.source_name is not None:
                source['name'] = sample_result.source_name
            data['source'] = source
        model_index_data.append(data)
    model_index = [{'name': model_name, 'results': model_index_data}]
    return _remove_none(model_index)