import os
import re
import subprocess
import sys
import unittest
from io import BytesIO
from io import UnsupportedOperation as _UnsupportedOperation
import iso8601
from testtools import ExtendedToOriginalDecorator, content, content_type
from testtools.compat import _b, _u
from testtools.content import TracebackContent
from testtools import CopyStreamResult, testresult
from subunit import chunked, details
from subunit.v2 import ByteStreamToStreamResult, StreamResultToBytes
def make_stream_binary(stream):
    """Ensure that a stream will be binary safe. See _make_binary_on_windows.
    
    :return: A binary version of the same stream (some streams cannot be
        'fixed' but can be unwrapped).
    """
    try:
        fileno = stream.fileno()
    except (_UnsupportedOperation, AttributeError):
        pass
    else:
        _make_binary_on_windows(fileno)
    return _unwrap_text(stream)