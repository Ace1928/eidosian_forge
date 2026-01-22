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
def mock_print_func(value, sep=' ', end='\n', file=sys.stdout, flush=False):
    values.append(value)
    if value == chr(55551):
        raise UnicodeEncodeError('utf-8', chr(55551), 0, 1, '')