from datetime import datetime
from io import StringIO
from pathlib import Path
import re
from shutil import get_terminal_size
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
from pandas.io.formats import printing
import pandas.io.formats.format as fmt
def test_east_asian_unicode_true(self):
    with option_context('display.unicode.east_asian_width', True):
        df = DataFrame({'a': ['あ', 'いいい', 'う', 'ええええええ'], 'b': [1, 222, 33333, 4]}, index=['a', 'bb', 'c', 'ddd'])
        expected = '                a      b\na              あ      1\nbb         いいい    222\nc              う  33333\nddd  ええええええ      4'
        assert repr(df) == expected
        df = DataFrame({'a': [1, 222, 33333, 4], 'b': ['あ', 'いいい', 'う', 'ええええええ']}, index=['a', 'bb', 'c', 'ddd'])
        expected = '         a             b\na        1            あ\nbb     222        いいい\nc    33333            う\nddd      4  ええええええ'
        assert repr(df) == expected
        df = DataFrame({'a': ['あああああ', 'い', 'う', 'えええ'], 'b': ['あ', 'いいい', 'う', 'ええええええ']}, index=['a', 'bb', 'c', 'ddd'])
        expected = '              a             b\na    あああああ            あ\nbb           い        いいい\nc            う            う\nddd      えええ  ええええええ'
        assert repr(df) == expected
        df = DataFrame({'b': ['あ', 'いいい', 'う', 'ええええええ'], 'あああああ': [1, 222, 33333, 4]}, index=['a', 'bb', 'c', 'ddd'])
        expected = '                b  あああああ\na              あ           1\nbb         いいい         222\nc              う       33333\nddd  ええええええ           4'
        assert repr(df) == expected
        df = DataFrame({'a': ['あああああ', 'い', 'う', 'えええ'], 'b': ['あ', 'いいい', 'う', 'ええええええ']}, index=['あああ', 'いいいいいい', 'うう', 'え'])
        expected = '                       a             b\nあああ        あああああ            あ\nいいいいいい          い        いいい\nうう                  う            う\nえ                えええ  ええええええ'
        assert repr(df) == expected
        df = DataFrame({'a': ['あああああ', 'い', 'う', 'えええ'], 'b': ['あ', 'いいい', 'う', 'ええええええ']}, index=Index(['あ', 'い', 'うう', 'え'], name='おおおお'))
        expected = '                   a             b\nおおおお                          \nあ        あああああ            あ\nい                い        いいい\nうう              う            う\nえ            えええ  ええええええ'
        assert repr(df) == expected
        df = DataFrame({'あああ': ['あああ', 'い', 'う', 'えええええ'], 'いいいいい': ['あ', 'いいい', 'う', 'ええ']}, index=Index(['あ', 'いいい', 'うう', 'え'], name='お'))
        expected = '            あああ いいいいい\nお                           \nあ          あああ         あ\nいいい          い     いいい\nうう            う         う\nえ      えええええ       ええ'
        assert repr(df) == expected
        idx = MultiIndex.from_tuples([('あ', 'いい'), ('う', 'え'), ('おおお', 'かかかか'), ('き', 'くく')])
        df = DataFrame({'a': ['あああああ', 'い', 'う', 'えええ'], 'b': ['あ', 'いいい', 'う', 'ええええええ']}, index=idx)
        expected = '                          a             b\nあ     いい      あああああ            あ\nう     え                い        いいい\nおおお かかかか          う            う\nき     くく          えええ  ええええええ'
        assert repr(df) == expected
        with option_context('display.max_rows', 3, 'display.max_columns', 3):
            df = DataFrame({'a': ['あああああ', 'い', 'う', 'えええ'], 'b': ['あ', 'いいい', 'う', 'ええええええ'], 'c': ['お', 'か', 'ききき', 'くくくくくく'], 'ああああ': ['さ', 'し', 'す', 'せ']}, columns=['a', 'b', 'c', 'ああああ'])
            expected = '             a  ... ああああ\n0   あああああ  ...       さ\n..         ...  ...      ...\n3       えええ  ...       せ\n\n[4 rows x 4 columns]'
            assert repr(df) == expected
            df.index = ['あああ', 'いいいい', 'う', 'aaa']
            expected = '                 a  ... ああああ\nあああ  あああああ  ...       さ\n...            ...  ...      ...\naaa         えええ  ...       せ\n\n[4 rows x 4 columns]'
            assert repr(df) == expected
        df = DataFrame({'b': ['あ', 'いいい', '¡¡', 'ええええええ'], 'あああああ': [1, 222, 33333, 4]}, index=['a', 'bb', 'c', '¡¡¡'])
        expected = '                b  あああああ\na              あ           1\nbb         いいい         222\nc              ¡¡       33333\n¡¡¡  ええええええ           4'
        assert repr(df) == expected