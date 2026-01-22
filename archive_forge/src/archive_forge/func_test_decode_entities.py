import logging
import unittest
import numpy as np
from gensim import utils
from gensim.test.utils import datapath, get_tmpfile
def test_decode_entities(self):
    body = u'It&#146;s the Year of the Horse. YES VIN DIESEL &#128588; &#128175;'
    expected = u'It\x92s the Year of the Horse. YES VIN DIESEL 🙌 💯'
    self.assertEqual(utils.decode_htmlentities(body), expected)