from __future__ import annotations
import codecs
import email.message
import ipaddress
import mimetypes
import os
import re
import time
import typing
from pathlib import Path
from urllib.request import getproxies
import sniffio
from ._types import PrimitiveData
def get_ca_bundle_from_env() -> str | None:
    if 'SSL_CERT_FILE' in os.environ:
        ssl_file = Path(os.environ['SSL_CERT_FILE'])
        if ssl_file.is_file():
            return str(ssl_file)
    if 'SSL_CERT_DIR' in os.environ:
        ssl_path = Path(os.environ['SSL_CERT_DIR'])
        if ssl_path.is_dir():
            return str(ssl_path)
    return None