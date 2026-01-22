import os
import sys
import tempfile
import torch
from .state import AcceleratorState, PartialState
from .utils import (

    Launches a training function using several processes on CPU for debugging purposes.

    <Tip warning={true}>

    This function is provided for internal testing and debugging, but it's not intended for real trainings. It will
    only use the CPU.

    </Tip>

    Args:
        function (`Callable`):
            The training function to execute.
        args (`Tuple`):
            Tuple of arguments to pass to the function (it will receive `*args`).
        num_processes (`int`, *optional*, defaults to 2):
            The number of processes to use for training.
    