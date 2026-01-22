from __future__ import annotations
import binascii
import re
import sys
import typing
import warnings
from cryptography import utils
from cryptography.hazmat.bindings._rust import x509 as rust_x509
from cryptography.x509.oid import NameOID, ObjectIdentifier

        Parses the `data` string and converts it to a Name.

        According to RFC4514 section 2.1 the RDNSequence must be
        reversed when converting to string representation. So, when
        we parse it, we need to reverse again to get the RDNs on the
        correct order.
        