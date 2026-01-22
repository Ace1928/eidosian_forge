import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_decodeSmallValue(self):
    input = '-9223372036854775808'
    ujson.decode(input)