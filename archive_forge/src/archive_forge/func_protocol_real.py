import os
import uuid
import xmltodict
from pytest import skip, fixture
from mock import patch
@fixture(scope='module')
def protocol_real():
    endpoint = os.environ.get('WINRM_ENDPOINT', None)
    transport = os.environ.get('WINRM_TRANSPORT', None)
    username = os.environ.get('WINRM_USERNAME', None)
    password = os.environ.get('WINRM_PASSWORD', None)
    if endpoint:
        settings = dict(endpoint=endpoint, operation_timeout_sec=5, read_timeout_sec=7)
        if transport:
            settings['transport'] = transport
        if username:
            settings['username'] = username
        if password:
            settings['password'] = password
        from winrm.protocol import Protocol
        protocol = Protocol(**settings)
        return protocol
    else:
        skip('WINRM_ENDPOINT environment variable was not set. Integration tests will be skipped')