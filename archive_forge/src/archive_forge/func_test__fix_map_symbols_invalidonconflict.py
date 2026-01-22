import pytest
import rpy2.robjects.packages_utils as p_u
def test__fix_map_symbols_invalidonconflict():
    msg_prefix = ''
    exception = ValueError
    symbol_mappings = {'foo': 'foo'}
    conflicts = {'foo_bar': ['foo.bar', 'foo_bar']}
    on_conflict = 'foo'
    with pytest.raises(ValueError):
        p_u._fix_map_symbols(symbol_mappings, conflicts, on_conflict, msg_prefix, exception)