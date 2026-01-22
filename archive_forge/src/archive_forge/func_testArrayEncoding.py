import datetime
import json
import math
import unittest
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import extra_types
def testArrayEncoding(self):
    array = [3, 'four', False]
    json_array = extra_types.JsonArray(entries=[extra_types.JsonValue(integer_value=3), extra_types.JsonValue(string_value='four'), extra_types.JsonValue(boolean_value=False)])
    self.assertRoundTrip(array)
    self.assertRoundTrip(json_array)
    self.assertTranslations(array, json_array)