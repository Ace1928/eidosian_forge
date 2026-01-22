import gettext
import os
import re
import textwrap
import warnings
from . import declarative
def is_validator(obj):
    """Check whether obj is a Validator instance or class."""
    return isinstance(obj, Validator) or (isinstance(obj, type) and issubclass(obj, Validator))