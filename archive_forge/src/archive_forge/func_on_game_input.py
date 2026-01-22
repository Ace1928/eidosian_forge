import numpy as np
from ai_logic import (
from gui_utils import (
from typing import List, Tuple
import types
import importlib.util
import logging
import random
@StandardDecorator()
def on_game_input(key: str):
    """
    Performs tasks in response to user input.

    Args:
        key (str): The key pressed by the user.
    """