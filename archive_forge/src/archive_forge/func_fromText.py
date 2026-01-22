import re
import sys
import string
import socket
from socket import AF_INET, AF_INET6
from typing import (
from unicodedata import normalize
from ._socket import inet_pton
from idna import encode as idna_encode, decode as idna_decode
@classmethod
def fromText(cls, s, lazy=False):
    return cls.from_text(s, lazy=lazy)