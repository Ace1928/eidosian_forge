from tests.unit import unittest
from boto.exception import BotoServerError, S3CreateError, JSONResponseError
from httpretty import HTTPretty, httprettified
def test_message_json_response_error(self):
    body = {'__type': 'com.amazon.coral.validate#ValidationException', 'message': 'The attempted filter operation is not supported for the provided filter argument count'}
    jre = JSONResponseError('400', 'Bad Request', body=body)
    self.assertEqual(jre.status, '400')
    self.assertEqual(jre.reason, 'Bad Request')
    self.assertEqual(jre.error_message, body['message'])
    self.assertEqual(jre.error_message, jre.message)
    self.assertEqual(jre.code, 'ValidationException')
    self.assertEqual(jre.code, jre.error_code)