import itertools
from typing import Union, Iterable, Dict, Iterator
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, group
def average_parameters_or_parameter_groups(params: Union[Iterable[torch.nn.Parameter], Iterable[Dict[str, torch.nn.Parameter]]], process_group: ProcessGroup):
    """
    Averages parameters of a model or parameter groups of an optimizer.
    """
    average_parameters(iter(get_params_to_average(params)), process_group)