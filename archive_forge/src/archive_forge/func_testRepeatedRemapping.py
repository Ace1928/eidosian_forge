import base64
import datetime
import json
import sys
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import util
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import extra_types
def testRepeatedRemapping(self):
    encoding.AddCustomJsonEnumMapping(MessageWithRemappings.SomeEnum, 'enum_value', 'wire_name')
    encoding.AddCustomJsonFieldMapping(MessageWithRemappings, 'double_encoding', 'doubleEncoding')
    encoding.AddCustomJsonFieldMapping(MessageWithRemappings, 'another_field', 'anotherField')
    encoding.AddCustomJsonFieldMapping(MessageWithRemappings, 'repeated_field', 'repeatedField')
    self.assertRaises(exceptions.InvalidDataError, encoding.AddCustomJsonFieldMapping, MessageWithRemappings, 'double_encoding', 'something_else')
    self.assertRaises(exceptions.InvalidDataError, encoding.AddCustomJsonFieldMapping, MessageWithRemappings, 'enum_field', 'anotherField')
    self.assertRaises(exceptions.InvalidDataError, encoding.AddCustomJsonEnumMapping, MessageWithRemappings.SomeEnum, 'enum_value', 'another_name')
    self.assertRaises(exceptions.InvalidDataError, encoding.AddCustomJsonEnumMapping, MessageWithRemappings.SomeEnum, 'second_value', 'wire_name')