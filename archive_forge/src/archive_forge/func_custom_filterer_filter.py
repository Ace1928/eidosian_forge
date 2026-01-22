from __future__ import (absolute_import, division, print_function)
import logging
import os
import re
import runpy
import sys
import warnings
def custom_filterer_filter(self, record):
    """Globally omit logging of unwanted messages."""
    if LOGGING_MESSAGE_FILTER.search(record.getMessage()):
        return 0
    return BUILTIN_FILTERER_FILTER(self, record)