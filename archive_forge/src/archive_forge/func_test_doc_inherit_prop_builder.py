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
def test_doc_inherit_prop_builder():

    def builder(name):
        return property(lambda self: name)

    class Parent:
        prop = builder('Parent')

    @modin.utils._inherit_docstrings(Parent)
    class Child(Parent):
        prop = builder('Child')
    assert Parent().prop == 'Parent'
    assert Child().prop == 'Child'