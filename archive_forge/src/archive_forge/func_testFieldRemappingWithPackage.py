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
def testFieldRemappingWithPackage(self):
    this_module = sys.modules[__name__]
    package_name = 'my_package'
    try:
        setattr(this_module, 'package', package_name)
        encoding.AddCustomJsonFieldMapping(MessageWithPackageAndRemappings, 'another_field', 'wire_field_name', package=package_name)
        msg = MessageWithPackageAndRemappings(another_field='my value')
        json_message = encoding.MessageToJson(msg)
        self.assertEqual('{"wire_field_name": "my value"}', json_message)
        self.assertEqual(msg, encoding.JsonToMessage(MessageWithPackageAndRemappings, json_message))
    finally:
        delattr(this_module, 'package')