import os
import json
import pathlib
from typing import Optional, Union, Dict, Any
from lazyops.types.models import BaseSettings, validator
from lazyops.types.classprops import lazyproperty
from lazyops.imports._fileio import (
@lazyproperty
@require_fileio()
def minio_bucket_path(self):
    if self.minio_bucket is None:
        return None
    bucket = self.minio_bucket
    if not bucket.startswith('minio://'):
        bucket = f'minio://{bucket}'
    return File(bucket)