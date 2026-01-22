import itertools
import platform
import sys
from abc import abstractmethod
from typing import Callable, List, Optional, Tuple, Union
import click
import wandb
from . import ipython, sparkline
def panel(self, columns: List[str]) -> str:
    row = ''.join([f'<div class="wandb-col">{col}</div>' for col in columns])
    return f'{ipython.TABLE_STYLES}<div class="wandb-row">{row}</div>'