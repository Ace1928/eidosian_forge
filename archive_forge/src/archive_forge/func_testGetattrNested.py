import unittest
from apitools.base.py import list_pager
from apitools.base.py.testing import mock
from samples.fusiontables_sample.fusiontables_v1 \
from samples.fusiontables_sample.fusiontables_v1 \
from samples.iam_sample.iam_v1 import iam_v1_client as iam_client
from samples.iam_sample.iam_v1 import iam_v1_messages as iam_messages
def testGetattrNested(self):
    o = Example()
    self.assertEqual(list_pager._GetattrNested(o, 'a'), 'aaa')
    self.assertEqual(list_pager._GetattrNested(o, ('a',)), 'aaa')
    o.b = Example()
    self.assertEqual(list_pager._GetattrNested(o, ('b', 'c')), 'ccc')