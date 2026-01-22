import base64
import re
from io import BytesIO
from urllib.request import parse_http_list, parse_keqv_list
from .. import errors, osutils, tests, transport
from ..bzr.smart import medium
from ..transport import chroot
from . import http_server
def authorized(self, user, password):
    """Check that the given user provided the right password"""
    expected_password = self.password_of.get(user, None)
    return expected_password is not None and password == expected_password