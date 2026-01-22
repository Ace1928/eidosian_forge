from abc import ABC, abstractmethod
from argparse import Namespace
from functools import wraps
from typing import Any, Callable, Dict, Optional, Union
from torch import Tensor
from torch.nn import Module
from lightning_fabric.utilities.rank_zero import rank_zero_only
def rank_zero_experiment(fn: Callable) -> Callable:
    """Returns the real experiment on rank 0 and otherwise the _DummyExperiment."""

    @wraps(fn)
    def experiment(self: Logger) -> Union[Any, _DummyExperiment]:
        """
        Note:
            ``self`` is a custom logger instance. The loggers typically wrap an ``experiment`` method
            with a ``@rank_zero_experiment`` decorator.

            ``Union[Any, _DummyExperiment]`` is used because the wrapped hooks have several return
            types that are specific to the custom logger. The return type here can be considered as
            ``Union[return type of logger.experiment, _DummyExperiment]``.
        """
        if rank_zero_only.rank > 0:
            return _DummyExperiment()
        return fn(self)
    return experiment