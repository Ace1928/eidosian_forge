from boto.mws.connection import MWSConnection, api_call_map, destructure_object
from boto.mws.response import (ResponseElement, GetFeedSubmissionListResult,
from boto.exception import BotoServerError
from tests.compat import unittest
from tests.unit import AWSMockServiceTestCase
from mock import MagicMock
def test_destructure_object(self):
    response = ResponseElement()
    response.C = 'four'
    response.D = 'five'
    inputs = [('A', 'B'), ['B', 'A'], set(['C']), False, 'String', {'A': 'one', 'B': 'two'}, response, {'A': 'one', 'B': 'two', 'C': [{'D': 'four', 'E': 'five'}, {'F': 'six', 'G': 'seven'}]}]
    outputs = [{'Prefix.1': 'A', 'Prefix.2': 'B'}, {'Prefix.1': 'B', 'Prefix.2': 'A'}, {'Prefix.1': 'C'}, {'Prefix': 'false'}, {'Prefix': 'String'}, {'Prefix.A': 'one', 'Prefix.B': 'two'}, {'Prefix.C': 'four', 'Prefix.D': 'five'}, {'Prefix.A': 'one', 'Prefix.B': 'two', 'Prefix.C.member.1.D': 'four', 'Prefix.C.member.1.E': 'five', 'Prefix.C.member.2.F': 'six', 'Prefix.C.member.2.G': 'seven'}]
    for user, amazon in zip(inputs, outputs):
        result = {}
        members = user is inputs[-1]
        destructure_object(user, result, prefix='Prefix', members=members)
        self.assertEqual(result, amazon)