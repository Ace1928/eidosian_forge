from botocore.exceptions import ClientError
from botocore.compat import six
from s3transfer.exceptions import RetriesExceededError as \
from s3transfer.manager import TransferConfig as S3TransferConfig
from s3transfer.manager import TransferManager
from s3transfer.futures import NonThreadedExecutor
from s3transfer.subscribers import BaseSubscriber
from s3transfer.utils import OSUtils
from boto3.exceptions import RetriesExceededError, S3UploadFailedError
def on_progress(self, bytes_transferred, **kwargs):
    self._callback(bytes_transferred)