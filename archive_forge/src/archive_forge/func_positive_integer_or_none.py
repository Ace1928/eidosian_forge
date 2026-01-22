import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
import sys
import os
from importlib import import_module
from .tpot import TPOTClassifier, TPOTRegressor
from ._version import __version__
def positive_integer_or_none(value):
    """Ensure that the provided value is a positive integer or None.

    Parameters
    ----------
    value: string
        The number to evaluate

    Returns
    -------
    value: int or None
        Returns a positive integer or None
    """
    if value.lower() == 'none':
        value = None
    else:
        try:
            value = int(value)
        except Exception:
            raise argparse.ArgumentTypeError("Invalid int value: '{}'".format(value))
        if value < 0:
            raise argparse.ArgumentTypeError("Invalid positive int value: '{}'".format(value))
    return value