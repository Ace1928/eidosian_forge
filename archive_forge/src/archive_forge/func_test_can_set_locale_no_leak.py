import codecs
import locale
import os
import pytest
from pandas._config.localization import (
from pandas.compat import ISMUSL
import pandas as pd
@pytest.mark.parametrize('lang,enc', [('it_CH', 'UTF-8'), ('en_US', 'ascii'), ('zh_CN', 'GB2312'), ('it_IT', 'ISO-8859-1')])
@pytest.mark.parametrize('lc_var', (locale.LC_ALL, locale.LC_CTYPE, locale.LC_TIME))
def test_can_set_locale_no_leak(lang, enc, lc_var):
    before_locale = _get_current_locale(lc_var)
    can_set_locale((lang, enc), locale.LC_ALL)
    after_locale = _get_current_locale(lc_var)
    assert before_locale == after_locale