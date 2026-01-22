from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Tuple
import torch
from torch import Tensor
@abstractmethod
def configure_response(self) -> Dict[str, Any]:
    """Returns a response to validate the server response."""