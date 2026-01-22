import enum
import logging
import os
import sys
import typing
import warnings
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import callbacks
def isinitialized() -> bool:
    """Is the embedded R initialized."""
    return bool(rpy2_embeddedR_isinitialized & RPY_R_Status.INITIALIZED.value)