import os
import math
import functools
import logging
import socket
import threading
import random
import string
import concurrent.futures
from botocore.compat import six
from botocore.vendored.requests.packages.urllib3.exceptions import \
from botocore.exceptions import IncompleteReadError
import s3transfer.compat
from s3transfer.exceptions import RetriesExceededError, S3UploadFailedError
def enable_upload_callbacks(request, operation_name, **kwargs):
    if operation_name in ['PutObject', 'UploadPart'] and hasattr(request.body, 'enable_callback'):
        request.body.enable_callback()