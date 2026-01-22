import os
import json
import pathlib
from typing import Optional, Union, Dict, Any
from lazyops.types.models import BaseSettings, validator
from lazyops.types.classprops import lazyproperty
from lazyops.imports._fileio import (
@lazyproperty
@require_fileio()
def s3_bucket_path(self):
    if self.s3_bucket is None:
        return None
    bucket = self.s3_bucket
    if not bucket.startswith('s3://'):
        bucket = f's3://{bucket}'
    return File(bucket)