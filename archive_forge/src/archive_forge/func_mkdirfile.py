import base64
import binascii
import concurrent.futures
import datetime
import hashlib
import json
import math
import os
import platform
import socket
import time
import urllib.parse
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple
import urllib3
from blobfile import _common as common
from blobfile._common import (
def mkdirfile(conf: Config, path: str) -> None:
    if not path.endswith('/'):
        path += '/'
    bucket, blob = split_path(path)
    req = Request(url=build_url('/upload/storage/v1/b/{bucket}/o', bucket=bucket), method='POST', params=dict(uploadType='media', name=blob), success_codes=(200, 400))
    resp = execute_api_request(conf, req)
    if resp.status == 400:
        raise Error(f"Unable to create directory, bucket does not exist: '{path}'")