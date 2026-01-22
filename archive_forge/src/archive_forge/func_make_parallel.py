import os
from typing import Union, Optional, Tuple, Any, List, Sized, TypeVar
import itertools
from collections import namedtuple
import parlai.utils.logging as logging
import torch.optim
def make_parallel(self, model: torch.nn.Module) -> torch.nn.Module:
    """
        Allocate specific layers in a model to be ModelParallel.

        Limited to only ModuleLists within the model.  Uses some heuristics to
        attempt to evenly distribute layers across GPUs, in order to balance
        memory usage. They are:

        - Assume the 0th GPU will host the optimizer, word embeddings, etc.
        - Assume activation memory is linear with the number of parameters.
        - All layers are approximately equal in size.
        """
    self.__device_allocations['cuda:0'] += trainable_parameters(model) * 3
    model.apply(self._place_modulelist)
    model._apply(self._move_rest_to_cuda0)
    return model