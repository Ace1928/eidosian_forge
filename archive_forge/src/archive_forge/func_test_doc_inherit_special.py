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
def test_doc_inherit_special(wrapped_cls):
    _check_doc(wrapped_cls.static, BaseChild.static)
    _check_doc(wrapped_cls.clsmtd, BaseChild.clsmtd)