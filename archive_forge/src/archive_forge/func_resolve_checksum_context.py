import base64
import io
import logging
from binascii import crc32
from hashlib import sha1, sha256
from botocore.compat import HAS_CRT
from botocore.exceptions import (
from botocore.response import StreamingBody
from botocore.utils import (
def resolve_checksum_context(request, operation_model, params):
    resolve_request_checksum_algorithm(request, operation_model, params)
    resolve_response_checksum_algorithms(request, operation_model, params)