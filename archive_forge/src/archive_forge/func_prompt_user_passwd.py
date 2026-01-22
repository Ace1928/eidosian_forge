import urllib.request
import base64
import bisect
import email
import hashlib
import http.client
import io
import os
import posixpath
import re
import socket
import string
import sys
import time
import tempfile
import contextlib
import warnings
from urllib.error import URLError, HTTPError, ContentTooShortError
from urllib.parse import (
from urllib.response import addinfourl, addclosehook
def prompt_user_passwd(self, host, realm):
    """Override this in a GUI environment!"""
    import getpass
    try:
        user = input('Enter username for %s at %s: ' % (realm, host))
        passwd = getpass.getpass('Enter password for %s in %s at %s: ' % (user, realm, host))
        return (user, passwd)
    except KeyboardInterrupt:
        print()
        return (None, None)