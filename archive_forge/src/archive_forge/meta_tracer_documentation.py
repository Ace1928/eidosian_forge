import torch
import torch.fx
import warnings
import functools
import builtins
from typing import Any, Callable, Dict, Optional, Union

        Helper method which tries to insert a module that was not declared as submodule.
        