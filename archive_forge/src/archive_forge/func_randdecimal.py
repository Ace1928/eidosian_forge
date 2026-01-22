import contextlib
import decimal
import gc
import numpy as np
import os
import random
import re
import shutil
import signal
import socket
import string
import subprocess
import sys
import time
import pytest
import pyarrow as pa
import pyarrow.fs
def randdecimal(precision, scale):
    """Generate a random decimal value with specified precision and scale.

    Parameters
    ----------
    precision : int
        The maximum number of digits to generate. Must be an integer between 1
        and 38 inclusive.
    scale : int
        The maximum number of digits following the decimal point.  Must be an
        integer greater than or equal to 0.

    Returns
    -------
    decimal_value : decimal.Decimal
        A random decimal.Decimal object with the specified precision and scale.
    """
    assert 1 <= precision <= 38, 'precision must be between 1 and 38 inclusive'
    if scale < 0:
        raise ValueError('randdecimal does not yet support generating decimals with negative scale')
    max_whole_value = 10 ** (precision - scale) - 1
    whole = random.randint(-max_whole_value, max_whole_value)
    if not scale:
        return decimal.Decimal(whole)
    max_fractional_value = 10 ** scale - 1
    fractional = random.randint(0, max_fractional_value)
    return decimal.Decimal('{}.{}'.format(whole, str(fractional).rjust(scale, '0')))