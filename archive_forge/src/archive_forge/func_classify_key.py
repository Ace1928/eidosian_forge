import operator
from ... import (branch, commands, config, errors, option, trace, tsort, ui,
from ...revision import NULL_REVISION
from .classify import classify_delta
def classify_key(item):
    """Sort key for item of (author, count) from classify_delta."""
    return (-item[1], item[0])