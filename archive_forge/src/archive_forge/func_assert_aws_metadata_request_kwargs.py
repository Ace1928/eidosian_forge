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
def assert_aws_metadata_request_kwargs(cls, request_kwargs, url, headers=None, method='GET'):
    assert request_kwargs['url'] == url
    assert request_kwargs['method'] == method
    if headers:
        assert request_kwargs['headers'] == headers
    else:
        assert 'headers' not in request_kwargs or request_kwargs['headers'] is None
    assert 'body' not in request_kwargs