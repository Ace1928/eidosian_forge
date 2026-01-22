import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_decodeBigEscape(self):
    for x in range(10):
        base = 'å'.encode('utf-8')
        quote = '"'.encode()
        input = quote + base * 1024 * 1024 * 2 + quote
        ujson.decode(input)