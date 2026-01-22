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
def parallel_upload(conf: Config, executor: concurrent.futures.Executor, src: str, dst: str, return_md5: bool) -> Optional[str]:
    with open(src, 'rb') as f:
        md5_digest = common.block_md5(f)
    hexdigest = binascii.hexlify(md5_digest).decode('utf8')
    s = os.stat(src)
    if s.st_size == 0:
        with StreamingWriteFile(conf, dst) as f:
            pass
        return hexdigest if return_md5 else None
    dstbucket, dstname = split_path(dst)
    source_objects = []
    object_names = []
    max_workers = getattr(executor, '_max_workers', os.cpu_count() or 1)
    part_size = max(math.ceil(s.st_size / max_workers), common.PARALLEL_COPY_MINIMUM_PART_SIZE)
    i = 0
    start = 0
    futures = []
    while start < s.st_size:
        suffix = f'.part.{i}'
        future = executor.submit(_upload_part, conf, src, start, min(part_size, s.st_size - start), dst + suffix)
        futures.append(future)
        object_names.append(dstname + suffix)
        i += 1
        start += part_size
    for name, future in zip(object_names, futures):
        generation = future.result()
        source_objects.append({'name': name, 'generation': generation, 'objectPreconditions': {'ifGenerationMatch': generation}})
    req = Request(url=build_url('/storage/v1/b/{destinationBucket}/o/{destinationObject}/compose', destinationBucket=dstbucket, destinationObject=dstname), method='POST', data={'sourceObjects': source_objects}, success_codes=(200,))
    resp = execute_api_request(conf, req)
    metadata = json.loads(resp.data)
    maybe_update_md5(conf, dst, metadata['generation'], hexdigest)
    delete_futures = []
    for name in object_names:
        future = executor.submit(_delete_part, conf, dstbucket, name)
        delete_futures.append(future)
    for future in delete_futures:
        future.result()
    return hexdigest if return_md5 else None