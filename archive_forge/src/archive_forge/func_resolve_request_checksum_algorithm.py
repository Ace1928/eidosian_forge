import base64
import io
import logging
from binascii import crc32
from hashlib import sha1, sha256
from botocore.compat import HAS_CRT
from botocore.exceptions import (
from botocore.response import StreamingBody
from botocore.utils import (
def resolve_request_checksum_algorithm(request, operation_model, params, supported_algorithms=None):
    http_checksum = operation_model.http_checksum
    algorithm_member = http_checksum.get('requestAlgorithmMember')
    if algorithm_member and algorithm_member in params:
        if supported_algorithms is None:
            supported_algorithms = _SUPPORTED_CHECKSUM_ALGORITHMS
        algorithm_name = params[algorithm_member].lower()
        if algorithm_name not in supported_algorithms:
            if not HAS_CRT and algorithm_name in _CRT_CHECKSUM_ALGORITHMS:
                raise MissingDependencyException(msg=f'Using {algorithm_name.upper()} requires an additional dependency. You will need to pip install botocore[crt] before proceeding.')
            raise FlexibleChecksumError(error_msg='Unsupported checksum algorithm: %s' % algorithm_name)
        location_type = 'header'
        if operation_model.has_streaming_input:
            if request['url'].startswith('https:'):
                location_type = 'trailer'
        algorithm = {'algorithm': algorithm_name, 'in': location_type, 'name': 'x-amz-checksum-%s' % algorithm_name}
        if algorithm['name'] in request['headers']:
            return
        checksum_context = request['context'].get('checksum', {})
        checksum_context['request_algorithm'] = algorithm
        request['context']['checksum'] = checksum_context
    elif operation_model.http_checksum_required or http_checksum.get('requestChecksumRequired'):
        checksum_context = request['context'].get('checksum', {})
        checksum_context['request_algorithm'] = 'conditional-md5'
        request['context']['checksum'] = checksum_context