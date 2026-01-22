import datetime
import errno
import functools
import hashlib
import json
import os
import re
import sys
import threading
import urllib.parse
import uuid
from oslo_utils import encodeutils
from oslo_utils import strutils
import prettytable
import wrapt
from glanceclient._i18n import _
from glanceclient import exc
def endpoint_version_from_url(endpoint, default_version=None):
    if endpoint:
        endpoint, version = strip_version(endpoint)
        return (endpoint, version or default_version)
    else:
        return (None, default_version)