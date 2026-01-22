from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.transform.dedup import duplicates, unique, conflicts, distinct, \
def test_duplicates_headerless_no_keys():
    """Removing the duplicates from an empty table without specifying which
    columns shouldn't be a problem.
    """
    table = []
    actual = duplicates(table)
    expect = []
    ieq(expect, actual)