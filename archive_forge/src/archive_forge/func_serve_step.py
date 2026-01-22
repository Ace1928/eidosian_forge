from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Tuple
import torch
from torch import Tensor
@abstractmethod
def serve_step(self, *args: Tensor, **kwargs: Tensor) -> Dict[str, Tensor]:
    """Returns the predictions of your model as a dictionary.

        .. code-block:: python

            def serve_step(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                return {"predictions": self(x)}

        Args:
            args: The output from de-serializer functions provided by the ``configure_serialization`` hook.
            kwargs: The keyword output of the de-serializer functions provided by the ``configure_serialization`` hook.

        Return:
            - ``dict`` - A dictionary with their associated tensors.

        """