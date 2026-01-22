import unittest
import six
from apitools.base.py.testing import mock
from samples.iam_sample.iam_v1 import iam_v1_client  # nopep8
from samples.iam_sample.iam_v1 import iam_v1_messages  # nopep8
def testMessagesModule(self):
    self.assertEquals(iam_v1_messages, iam_v1_client.IamV1.MESSAGES_MODULE)