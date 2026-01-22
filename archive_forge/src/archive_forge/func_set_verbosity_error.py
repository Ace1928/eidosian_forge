import logging
import os
from logging import (
from typing import Optional
def set_verbosity_error():
    """
    Sets the verbosity to `logging.ERROR`.
    """
    return set_verbosity(ERROR)