import pickle
import re
import sys
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
def testAlternateVariant(self):
    """Test that default variant is used when not set."""
    field = messages.IntegerField(1, variant=messages.Variant.UINT32)
    self.assertEquals(messages.Variant.UINT32, field.variant)