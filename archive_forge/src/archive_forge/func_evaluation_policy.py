import os
import pytest
import sys
import textwrap
import unittest
from contextlib import contextmanager
from traitlets.config.loader import Config
from IPython import get_ipython
from IPython.core import completer
from IPython.utils.tempdir import TemporaryDirectory, TemporaryWorkingDirectory
from IPython.utils.generics import complete_object
from IPython.testing import decorators as dec
from IPython.core.completer import (
@contextmanager
def evaluation_policy(evaluation: str):
    ip = get_ipython()
    evaluation_original = ip.Completer.evaluation
    try:
        ip.Completer.evaluation = evaluation
        yield
    finally:
        ip.Completer.evaluation = evaluation_original