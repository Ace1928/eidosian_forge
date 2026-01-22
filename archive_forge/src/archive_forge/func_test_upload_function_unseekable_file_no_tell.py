import tempfile
import shutil
import os
import socket
from boto.compat import json
from boto.awslambda.layer1 import AWSLambdaConnection
from tests.unit import AWSMockServiceTestCase
from tests.compat import mock
def test_upload_function_unseekable_file_no_tell(self):
    sock = socket.socket()
    with self.assertRaises(TypeError):
        self.service_connection.upload_function(function_name='my-function', function_zip=sock, role='myrole', handler='myhandler', mode='event', runtime='nodejs')