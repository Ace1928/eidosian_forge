import pytest
import pandas as pd
from pandas import MultiIndex
import pandas._testing as tm
def test_changing_names(idx):
    assert [level.name for level in idx.levels] == ['first', 'second']
    view = idx.view()
    copy = idx.copy()
    shallow_copy = idx._view()
    new_names = [name + 'a' for name in idx.names]
    idx.names = new_names
    check_level_names(idx, ['firsta', 'seconda'])
    check_level_names(view, ['first', 'second'])
    check_level_names(copy, ['first', 'second'])
    check_level_names(shallow_copy, ['first', 'second'])
    shallow_copy.names = [name + 'c' for name in shallow_copy.names]
    check_level_names(idx, ['firsta', 'seconda'])