from __future__ import absolute_import, print_function, division
import logging
import pytest
import petl as etl
from petl.transform.validation import validate
from petl.test.helpers import ieq
from petl.errors import FieldSelectionError
def test_non_optional_constraint_with_missing_field():
    constraints = [dict(name='C1', field='foo', test=int)]
    table = (('bar', 'baz'), ('1999-99-99', 'z'))
    actual = validate(table, constraints)
    with pytest.raises(FieldSelectionError):
        debug(actual)