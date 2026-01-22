import numpy as np
from ai_logic import (
from gui_utils import (
from typing import List, Tuple
import types
import importlib.util
import logging
import random
@StandardDecorator()
def on_game_exit():
    """
    Performs cleanup tasks when the game is exited.
    """