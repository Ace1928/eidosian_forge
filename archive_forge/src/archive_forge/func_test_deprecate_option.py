import pytest
from pandas._config import config as cf
from pandas._config.config import OptionError
import pandas as pd
import pandas._testing as tm
def test_deprecate_option(self):
    cf.deprecate_option('foo')
    assert cf._is_deprecated('foo')
    with tm.assert_produces_warning(FutureWarning, match='deprecated'):
        with pytest.raises(KeyError, match="No such keys.s.: 'foo'"):
            cf.get_option('foo')
    cf.register_option('a', 1, 'doc', validator=cf.is_int)
    cf.register_option('b.c', 'hullo', 'doc2')
    cf.register_option('foo', 'hullo', 'doc2')
    cf.deprecate_option('a', removal_ver='nifty_ver')
    with tm.assert_produces_warning(FutureWarning, match='eprecated.*nifty_ver'):
        cf.get_option('a')
        msg = "Option 'a' has already been defined as deprecated"
        with pytest.raises(OptionError, match=msg):
            cf.deprecate_option('a')
    cf.deprecate_option('b.c', 'zounds!')
    with tm.assert_produces_warning(FutureWarning, match='zounds!'):
        cf.get_option('b.c')
    cf.register_option('d.a', 'foo', 'doc2')
    cf.register_option('d.dep', 'bar', 'doc2')
    assert cf.get_option('d.a') == 'foo'
    assert cf.get_option('d.dep') == 'bar'
    cf.deprecate_option('d.dep', rkey='d.a')
    with tm.assert_produces_warning(FutureWarning, match='eprecated'):
        assert cf.get_option('d.dep') == 'foo'
    with tm.assert_produces_warning(FutureWarning, match='eprecated'):
        cf.set_option('d.dep', 'baz')
    with tm.assert_produces_warning(FutureWarning, match='eprecated'):
        assert cf.get_option('d.dep') == 'baz'