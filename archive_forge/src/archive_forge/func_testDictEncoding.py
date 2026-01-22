import datetime
import json
import math
import unittest
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import extra_types
def testDictEncoding(self):
    d = {'a': 6, 'b': 'eleventeen'}
    json_d = extra_types.JsonObject(properties=[extra_types.JsonObject.Property(key='a', value=extra_types.JsonValue(integer_value=6)), extra_types.JsonObject.Property(key='b', value=extra_types.JsonValue(string_value='eleventeen'))])
    self.assertRoundTrip(d)
    translated_properties = extra_types._PythonValueToJsonProto(d).properties
    for p in json_d.properties:
        self.assertIn(p, translated_properties)
    for p in translated_properties:
        self.assertIn(p, json_d.properties)