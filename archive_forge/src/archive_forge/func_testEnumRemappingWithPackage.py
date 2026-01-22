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
def testEnumRemappingWithPackage(self):
    this_module = sys.modules[__name__]
    package_name = 'my_package'
    try:
        setattr(this_module, 'package', package_name)
        encoding.AddCustomJsonEnumMapping(MessageWithPackageAndRemappings.SomeEnum, 'enum_value', 'other_wire_name', package=package_name)
        msg = MessageWithPackageAndRemappings(enum_field=MessageWithPackageAndRemappings.SomeEnum.enum_value)
        json_message = encoding.MessageToJson(msg)
        self.assertEqual('{"enum_field": "other_wire_name"}', json_message)
        self.assertEqual(msg, encoding.JsonToMessage(MessageWithPackageAndRemappings, json_message))
    finally:
        delattr(this_module, 'package')