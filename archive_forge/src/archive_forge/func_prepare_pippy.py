import math
from types import MethodType
from typing import Any, Dict, List, Optional, Tuple, Union
from .state import PartialState
from .utils import (
def prepare_pippy(model, split_points: Optional[Union[str, List[str]]]='auto', no_split_module_classes: Optional[List[str]]=None, example_args: Optional[Tuple[Any]]=(), example_kwargs: Optional[Dict[str, Any]]=None, num_chunks: Optional[int]=None, gather_output: Optional[bool]=False):
    """
    Wraps `model` for pipeline parallel inference.

    Args:
        model (`torch.nn.Module`):
            A model we want to split for pipeline-parallel inference
        split_points (`str` or `List[str]`, defaults to 'auto'):
            How to generate the split points and chunk the model across each GPU. 'auto' will find the best balanced
            split given any model. Should be a list of layer names in the model to split by otherwise.
        no_split_module_classes (`List[str]`):
            A list of class names for layers we don't want to be split.
        example_args (tuple of model inputs):
            The expected inputs for the model that uses order-based inputs. Recommended to use this method if possible.
        example_kwargs (dict of model inputs)
            The expected inputs for the model that uses dictionary-based inputs. This is a *highly* limiting structure
            that requires the same keys be present at *all* inference calls. Not recommended unless the prior condition
            is true for all cases.
        num_chunks (`int`, defaults to the number of available GPUs):
            The number of different stages the Pipeline will have. By default it will assign one chunk per GPU, but
            this can be tuned and played with. In general one should have num_chunks >= num_gpus.
        gather_output (`bool`, defaults to `False`):
            If `True`, the output from the last GPU (which holds the true outputs) is sent across to all GPUs.
    """
    if not is_pippy_available():
        raise ImportError('`pippy` was not found to be installed on your system. Please install using `pip install torchpippy` or ensure you have at least version 0.2.0')
    state = PartialState()
    example_args = send_to_device(example_args, 'cpu')
    example_kwargs = send_to_device(example_kwargs, 'cpu')
    if num_chunks is None:
        num_chunks = state.num_processes
    if split_points == 'auto':
        device_map = generate_device_map(model, num_chunks, no_split_module_classes=no_split_module_classes)
        split_points = []
        for i in range(1, num_chunks):
            split_points.append(next((k for k, v in device_map.items() if v == i)))
    model.hf_split_points = split_points
    stage = build_pipeline(model, split_points, example_args, example_kwargs, num_chunks)
    model._original_forward = model.forward
    model._original_call = model.__call__
    model.pippy_stage = stage
    model.hf_split_points = split_points

    def forward(*args, **kwargs):
        return pippy_forward(stage.forward, num_chunks, gather_output, *args, **kwargs)
    model_forward = MethodType(forward, model)
    forward.__wrapped__ = model_forward
    model.forward = forward
    return model