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
def test_user_expression():
    ip.display_formatter.active_types = ip.display_formatter.format_types
    query = {'a': '1 + 2', 'b': '1/0'}
    r = ip.user_expressions(query)
    import pprint
    pprint.pprint(r)
    assert set(r.keys()) == set(query.keys())
    a = r['a']
    assert {'status', 'data', 'metadata'} == set(a.keys())
    assert a['status'] == 'ok'
    data = a['data']
    metadata = a['metadata']
    assert data.get('text/plain') == '3'
    b = r['b']
    assert b['status'] == 'error'
    assert b['ename'] == 'ZeroDivisionError'
    ip.display_formatter.active_types = ['text/plain']