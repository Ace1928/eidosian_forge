import datetime
import json
import os
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import _helpers
from google.auth import aws
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import transport
@classmethod
def make_serialized_aws_signed_request(cls, aws_security_credentials, region_name='us-east-2', url='https://sts.us-east-2.amazonaws.com?Action=GetCallerIdentity&Version=2011-06-15'):
    """Utility to generate serialize AWS signed requests.
        This makes it easy to assert generated subject tokens based on the
        provided AWS security credentials, regions and AWS STS endpoint.
        """
    request_signer = aws.RequestSigner(region_name)
    signed_request = request_signer.get_request_options(aws_security_credentials, url, 'POST')
    reformatted_signed_request = {'url': signed_request.get('url'), 'method': signed_request.get('method'), 'headers': [{'key': 'Authorization', 'value': signed_request.get('headers').get('Authorization')}, {'key': 'host', 'value': signed_request.get('headers').get('host')}, {'key': 'x-amz-date', 'value': signed_request.get('headers').get('x-amz-date')}]}
    if 'security_token' in aws_security_credentials:
        reformatted_signed_request.get('headers').append({'key': 'x-amz-security-token', 'value': signed_request.get('headers').get('x-amz-security-token')})
    (reformatted_signed_request.get('headers').append({'key': 'x-goog-cloud-target-resource', 'value': AUDIENCE}),)
    return urllib.parse.quote(json.dumps(reformatted_signed_request, separators=(',', ':'), sort_keys=True))