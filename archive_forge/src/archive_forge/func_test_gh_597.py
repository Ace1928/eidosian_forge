import asyncio
import ast
import os
import signal
import shutil
import sys
import tempfile
import unittest
import pytest
from unittest import mock
from os.path import join
from IPython.core.error import InputRejected
from IPython.core.inputtransformer import InputTransformer
from IPython.core import interactiveshell
from IPython.core.oinspect import OInfo
from IPython.testing.decorators import (
from IPython.testing import tools as tt
from IPython.utils.process import find_cmd
import warnings
import warnings
def test_gh_597(self):
    """Pretty-printing lists of objects with non-ascii reprs may cause
        problems."""

    class Spam(object):

        def __repr__(self):
            return 'é' * 50
    import IPython.core.formatters
    f = IPython.core.formatters.PlainTextFormatter()
    f([Spam(), Spam()])