import gzip
import hashlib
import io
import logging
import os
import re
import socket
import sys
import time
import urllib
from googlecloudsdk.core.util import encoding
from googlecloudsdk.third_party.appengine._internal import six_subset
def can_validate_certs():
    """Return True if we have the SSL package and can validate certificates."""
    return _CAN_VALIDATE_CERTS