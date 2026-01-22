from boto.mws.connection import MWSConnection, api_call_map, destructure_object
from boto.mws.response import (ResponseElement, GetFeedSubmissionListResult,
from boto.exception import BotoServerError
from tests.compat import unittest
from tests.unit import AWSMockServiceTestCase
from mock import MagicMock
def test_response_factory(self):
    connection = self.service_connection
    body = self.default_body()
    action = 'GetFeedSubmissionList'
    parser = connection._response_factory(action, connection=connection)
    response = connection._parse_response(parser, 'text/xml', body)
    self.assertEqual(response._action, action)
    self.assertEqual(response.__class__.__name__, action + 'Response')
    self.assertEqual(response._result.__class__, GetFeedSubmissionListResult)

    class MyResult(GetFeedSubmissionListResult):
        _hello = '_world'
    scope = {'GetFeedSubmissionListResult': MyResult}
    connection._setup_factories([scope])
    parser = connection._response_factory(action, connection=connection)
    response = connection._parse_response(parser, 'text/xml', body)
    self.assertEqual(response._action, action)
    self.assertEqual(response.__class__.__name__, action + 'Response')
    self.assertEqual(response._result.__class__, MyResult)
    self.assertEqual(response._result._hello, '_world')
    self.assertEqual(response._result.HasNext, 'true')