import os
import logging
from typing import Optional, List, Tuple
from ray._private.accelerators.accelerator import AcceleratorManager
Get the name of first Intel GPU. (supposed only one GPU type on a node)
        Example:
            name: 'Intel(R) Data Center GPU Max 1550'
            return name: 'Intel-GPU-Max-1550'
        Returns:
            A string representing the name of Intel GPU type.
        