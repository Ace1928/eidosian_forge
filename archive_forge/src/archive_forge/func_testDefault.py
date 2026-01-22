import datetime
import json
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.protorpclite import test_util
def testDefault(self):
    self.assertTrue(protojson.ProtoJson.get_default(), protojson.ProtoJson.get_default())
    instance = CustomProtoJson()
    protojson.ProtoJson.set_default(instance)
    self.assertTrue(instance is protojson.ProtoJson.get_default())