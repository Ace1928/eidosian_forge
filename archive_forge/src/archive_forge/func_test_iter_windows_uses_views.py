import logging
import unittest
import numpy as np
from gensim import utils
from gensim.test.utils import datapath, get_tmpfile
def test_iter_windows_uses_views(self):
    texts = [np.array(['this', 'is', 'a'], dtype='object'), ['test', 'document']]
    windows = list(utils.iter_windows(texts, 2))
    list_windows = [list(iterable) for iterable in windows]
    expected = [['this', 'is'], ['is', 'a'], ['test', 'document']]
    self.assertListEqual(list_windows, expected)
    windows[0][0] = 'modified'
    self.assertEqual('modified', texts[0][0])