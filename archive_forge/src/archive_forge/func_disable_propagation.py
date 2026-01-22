import logging
import os
from logging import (
from typing import Optional
def disable_propagation() -> None:
    """
    Disable propagation of the library log outputs. Note that log propagation is
    disabled by default.
    """
    _get_library_root_logger().propagate = False