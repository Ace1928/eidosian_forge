import json
from textwrap import dedent, indent
from unittest.mock import Mock, patch
import numpy as np
import pandas
import pytest
import modin.pandas as pd
import modin.utils
from modin.error_message import ErrorMessage
from modin.tests.pandas.utils import create_test_dfs
def test_doc_inherit_methods(wrapped_cls):
    _check_doc(wrapped_cls.method, BaseChild.method)
    _check_doc(wrapped_cls.base_method, BaseParent.base_method)
    _check_doc(wrapped_cls.own_method, BaseChild.own_method)
    assert wrapped_cls.no_overwrite.__doc__ != BaseChild.no_overwrite.__doc__
    assert not getattr(wrapped_cls.no_overwrite, '__doc_inherited__', False)