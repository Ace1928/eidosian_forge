import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
import wandb
def make_predictions(self, predict_fn: Callable) -> Union[Sequence, Dict[str, Sequence]]:
    """Produce predictions by passing `validation_inputs` to `predict_fn`.

        Args:
            predict_fn (Callable): Any function which can accept `validation_inputs` and produce
                a list of vectors or dictionary of lists of vectors

        Returns:
            (Sequence | Dict[str, Sequence]): The returned value of predict_fn
        """
    return predict_fn(self.validation_inputs)