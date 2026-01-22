from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Tuple
import torch
from torch import Tensor
@abstractmethod
def configure_serialization(self) -> Tuple[Dict[str, Callable], Dict[str, Callable]]:
    """Returns a tuple of dictionaries.

        The first dictionary contains the name of the ``serve_step`` input variables name as its keys
        and the associated de-serialization function (e.g function to convert a payload to tensors).

        The second dictionary contains the name of the ``serve_step`` output variables name as its keys
        and the associated serialization function (e.g function to convert a tensors into payload).

        """