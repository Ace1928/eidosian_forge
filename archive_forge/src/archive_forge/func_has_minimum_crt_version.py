import logging
import threading
from os import PathLike, fspath, getpid
from botocore.compat import HAS_CRT
from botocore.exceptions import ClientError
from s3transfer.exceptions import (
from s3transfer.futures import NonThreadedExecutor
from s3transfer.manager import TransferConfig as S3TransferConfig
from s3transfer.manager import TransferManager
from s3transfer.subscribers import BaseSubscriber
from s3transfer.utils import OSUtils
import boto3.s3.constants as constants
from boto3.exceptions import RetriesExceededError, S3UploadFailedError
def has_minimum_crt_version(minimum_version):
    """Not intended for use outside boto3."""
    if not HAS_CRT:
        return False
    crt_version_str = awscrt.__version__
    try:
        crt_version_ints = map(int, crt_version_str.split('.'))
        crt_version_tuple = tuple(crt_version_ints)
    except (TypeError, ValueError):
        return False
    return crt_version_tuple >= minimum_version