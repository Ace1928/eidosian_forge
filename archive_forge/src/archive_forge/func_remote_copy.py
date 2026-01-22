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
def remote_copy(conf: Config, src: str, dst: str, return_md5: bool) -> Optional[str]:
    srcbucket, srcname = split_path(src)
    dstbucket, dstname = split_path(dst)
    params = {}
    while True:
        req = Request(url=build_url('/storage/v1/b/{sourceBucket}/o/{sourceObject}/rewriteTo/b/{destinationBucket}/o/{destinationObject}', sourceBucket=srcbucket, sourceObject=srcname, destinationBucket=dstbucket, destinationObject=dstname), method='POST', params=params, success_codes=(200, 404))
        resp = execute_api_request(conf, req)
        if resp.status == 404:
            raise FileNotFoundError(f"Source file not found: '{src}'")
        result = json.loads(resp.data)
        if result['done']:
            if return_md5:
                return get_md5(result['resource'])
            else:
                return
        params['rewriteToken'] = result['rewriteToken']