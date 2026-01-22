import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
@pytest.mark.skipif(using_pyarrow_string_dtype(), reason='Change once infer_string is set to True by default')
def test_unicode_print(self):
    c = Categorical(['aaaaa', 'bb', 'cccc'] * 20)
    expected = "['aaaaa', 'bb', 'cccc', 'aaaaa', 'bb', ..., 'bb', 'cccc', 'aaaaa', 'bb', 'cccc']\nLength: 60\nCategories (3, object): ['aaaaa', 'bb', 'cccc']"
    assert repr(c) == expected
    c = Categorical(['ああああ', 'いいいいい', 'ううううううう'] * 20)
    expected = "['ああああ', 'いいいいい', 'ううううううう', 'ああああ', 'いいいいい', ..., 'いいいいい', 'ううううううう', 'ああああ', 'いいいいい', 'ううううううう']\nLength: 60\nCategories (3, object): ['ああああ', 'いいいいい', 'ううううううう']"
    assert repr(c) == expected
    with option_context('display.unicode.east_asian_width', True):
        c = Categorical(['ああああ', 'いいいいい', 'ううううううう'] * 20)
        expected = "['ああああ', 'いいいいい', 'ううううううう', 'ああああ', 'いいいいい', ..., 'いいいいい', 'ううううううう', 'ああああ', 'いいいいい', 'ううううううう']\nLength: 60\nCategories (3, object): ['ああああ', 'いいいいい', 'ううううううう']"
        assert repr(c) == expected