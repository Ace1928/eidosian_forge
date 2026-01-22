import unittest
from apitools.base.py import list_pager
from apitools.base.py.testing import mock
from samples.fusiontables_sample.fusiontables_v1 \
from samples.fusiontables_sample.fusiontables_v1 \
from samples.iam_sample.iam_v1 import iam_v1_client as iam_client
from samples.iam_sample.iam_v1 import iam_v1_messages as iam_messages
def testYieldFromListPaging(self):
    self.mocked_client.column.List.Expect(messages.FusiontablesColumnListRequest(maxResults=5, pageToken=None, tableId='mytable'), messages.ColumnList(items=[messages.Column(name='c0'), messages.Column(name='c1'), messages.Column(name='c2'), messages.Column(name='c3'), messages.Column(name='c4')], nextPageToken='x'))
    self.mocked_client.column.List.Expect(messages.FusiontablesColumnListRequest(maxResults=4, pageToken='x', tableId='mytable'), messages.ColumnList(items=[messages.Column(name='c5'), messages.Column(name='c6'), messages.Column(name='c7'), messages.Column(name='c8')]))
    client = fusiontables.FusiontablesV1(get_credentials=False)
    request = messages.FusiontablesColumnListRequest(tableId='mytable')
    results = list_pager.YieldFromList(client.column, request, limit=9, batch_size=5)
    self._AssertInstanceSequence(results, 9)