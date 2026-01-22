import logging
import os
from logging import (
from typing import Optional
def set_verbosity_debug():
    """
    Sets the verbosity to `logging.DEBUG`.
    """
    return set_verbosity(DEBUG)