import sys
import logging; log = logging.getLogger(__name__)
from types import ModuleType
def get_method_function(func):
    """given (potential) method, return underlying function"""
    return getattr(func, method_function_attr, func)