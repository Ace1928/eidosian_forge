import string
import pandas._config.config as cf
from pandas.io.formats import printing
def test_adjoin_unicode(self):
    data = [['あ', 'b', 'c'], ['dd', 'ええ', 'ff'], ['ggg', 'hhh', 'いいい']]
    expected = 'あ  dd  ggg\nb  ええ  hhh\nc  ff  いいい'
    adjoined = printing.adjoin(2, *data)
    assert adjoined == expected
    adj = printing._EastAsianTextAdjustment()
    expected = 'あ  dd    ggg\nb   ええ  hhh\nc   ff    いいい'
    adjoined = adj.adjoin(2, *data)
    assert adjoined == expected
    cols = adjoined.split('\n')
    assert adj.len(cols[0]) == 13
    assert adj.len(cols[1]) == 13
    assert adj.len(cols[2]) == 16
    expected = 'あ       dd         ggg\nb        ええ       hhh\nc        ff         いいい'
    adjoined = adj.adjoin(7, *data)
    assert adjoined == expected
    cols = adjoined.split('\n')
    assert adj.len(cols[0]) == 23
    assert adj.len(cols[1]) == 23
    assert adj.len(cols[2]) == 26