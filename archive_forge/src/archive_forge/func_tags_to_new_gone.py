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
def tags_to_new_gone(tags):
    """Split a list of tags into a new_set and a gone_set."""
    new_tags = set()
    gone_tags = set()
    for tag in tags:
        if tag[0] == '-':
            gone_tags.add(tag[1:])
        else:
            new_tags.add(tag)
    return (new_tags, gone_tags)