import re
from inspect import signature
from typing import Optional
import pytest
from sklearn.experimental import (
from sklearn.utils.discovery import all_displays, all_estimators, all_functions
def filter_errors(errors, method, Klass=None):
    """
    Ignore some errors based on the method type.

    These rules are specific for scikit-learn."""
    for code, message in errors:
        if code in ['RT02', 'GL01', 'GL02']:
            continue
        if code in ('PR02', 'GL08') and Klass is not None and (method is not None):
            method_obj = getattr(Klass, method)
            if isinstance(method_obj, property):
                continue
        if method is not None and code in ['EX01', 'SA01', 'ES01']:
            continue
        yield (code, message)