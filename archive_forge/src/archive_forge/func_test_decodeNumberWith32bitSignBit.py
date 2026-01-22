import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_decodeNumberWith32bitSignBit(self):
    docs = ('{"id": 3590016419}', '{"id": %s}' % 2 ** 31, '{"id": %s}' % 2 ** 32, '{"id": %s}' % (2 ** 32 - 1))
    results = (3590016419, 2 ** 31, 2 ** 32, 2 ** 32 - 1)
    for doc, result in zip(docs, results):
        self.assertEqual(ujson.decode(doc)['id'], result)