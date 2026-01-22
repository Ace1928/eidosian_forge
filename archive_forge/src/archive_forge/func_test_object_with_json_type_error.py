import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_object_with_json_type_error(self):
    for return_value in (None, 1234, 12.34, True, {}):

        class JSONTest:

            def __json__(self):
                return return_value
        d = {u'key': JSONTest()}
        self.assertRaises(TypeError, ujson.encode, d)