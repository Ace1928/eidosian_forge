import functools
import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union
import numpy as np
from transformers.utils import is_tf_available, is_torch_available
from .normalized_config import (

        Pads an input either to the desired length, or by a padding length.

        Args:
            input_:
                The tensor to pad.
            dim (`int`):
                The dimension along which to pad.
            desired_length (`Optional[int]`, defaults to `None`):
                The desired length along the dimension after padding.
            padding_length (`Optional[int]`, defaults to `None`):
                The length to pad along the dimension.
            value (`Union[int, float]`, defaults to 1):
                The value to use for padding.
            dtype (`Optional[Any]`, defaults to `None`):
                The dtype of the padding.

        Returns:
            The padded tensor.
        