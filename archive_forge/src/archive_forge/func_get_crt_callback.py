import logging
import threading
from io import BytesIO
import awscrt.http
import awscrt.s3
import botocore.awsrequest
import botocore.session
from awscrt.auth import (
from awscrt.io import (
from awscrt.s3 import S3Client, S3RequestTlsMode, S3RequestType
from botocore import UNSIGNED
from botocore.compat import urlsplit
from botocore.config import Config
from botocore.exceptions import NoCredentialsError
from s3transfer.constants import MB
from s3transfer.exceptions import TransferNotDoneError
from s3transfer.futures import BaseTransferFuture, BaseTransferMeta
from s3transfer.utils import (
def get_crt_callback(self, future, callback_type, before_subscribers=None, after_subscribers=None):

    def invoke_all_callbacks(*args, **kwargs):
        callbacks_list = []
        if before_subscribers is not None:
            callbacks_list += before_subscribers
        callbacks_list += get_callbacks(future, callback_type)
        if after_subscribers is not None:
            callbacks_list += after_subscribers
        for callback in callbacks_list:
            if callback_type == 'progress':
                callback(bytes_transferred=args[0])
            else:
                callback(*args, **kwargs)
    return invoke_all_callbacks