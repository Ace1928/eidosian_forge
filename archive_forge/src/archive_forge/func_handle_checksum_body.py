import base64
import io
import logging
from binascii import crc32
from hashlib import sha1, sha256
from botocore.compat import HAS_CRT
from botocore.exceptions import (
from botocore.response import StreamingBody
from botocore.utils import (
def handle_checksum_body(http_response, response, context, operation_model):
    headers = response['headers']
    checksum_context = context.get('checksum', {})
    algorithms = checksum_context.get('response_algorithms')
    if not algorithms:
        return
    for algorithm in algorithms:
        header_name = 'x-amz-checksum-%s' % algorithm
        if header_name not in headers:
            continue
        if '-' in headers[header_name]:
            continue
        if operation_model.has_streaming_output:
            response['body'] = _handle_streaming_response(http_response, response, algorithm)
        else:
            response['body'] = _handle_bytes_response(http_response, response, algorithm)
        checksum_context = response['context'].get('checksum', {})
        checksum_context['response_algorithm'] = algorithm
        response['context']['checksum'] = checksum_context
        return
    logger.info(f'Skipping checksum validation. Response did not contain one of the following algorithms: {algorithms}.')