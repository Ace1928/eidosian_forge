from botocore.exceptions import ClientError
from boto3.s3.transfer import create_transfer_manager
from boto3.s3.transfer import TransferConfig, S3Transfer
from boto3.s3.transfer import ProgressCallbackInvoker
from boto3 import utils
def bucket_download_fileobj(self, Key, Fileobj, ExtraArgs=None, Callback=None, Config=None):
    """Download an object from this bucket to a file-like-object.

    The file-like object must be in binary mode.

    This is a managed transfer which will perform a multipart download in
    multiple threads if necessary.

    Usage::

        import boto3
        s3 = boto3.resource('s3')
        bucket = s3.Bucket('mybucket')

        with open('filename', 'wb') as data:
            bucket.download_fileobj('mykey', data)

    :type Fileobj: a file-like object
    :param Fileobj: A file-like object to download into. At a minimum, it must
        implement the `write` method and must accept bytes.

    :type Key: str
    :param Key: The name of the key to download from.

    :type ExtraArgs: dict
    :param ExtraArgs: Extra arguments that may be passed to the
        client operation.

    :type Callback: function
    :param Callback: A method which takes a number of bytes transferred to
        be periodically called during the download.

    :type Config: boto3.s3.transfer.TransferConfig
    :param Config: The transfer configuration to be used when performing the
        download.
    """
    return self.meta.client.download_fileobj(Bucket=self.name, Key=Key, Fileobj=Fileobj, ExtraArgs=ExtraArgs, Callback=Callback, Config=Config)