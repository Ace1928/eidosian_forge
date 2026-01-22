import math
from botocore.compat import six
from s3transfer.compat import seekable, readable
from s3transfer.futures import IN_MEMORY_UPLOAD_TAG
from s3transfer.tasks import Task
from s3transfer.tasks import SubmissionTask
from s3transfer.tasks import CreateMultipartUploadTask
from s3transfer.tasks import CompleteMultipartUploadTask
from s3transfer.utils import get_callbacks
from s3transfer.utils import get_filtered_dict
from s3transfer.utils import DeferredOpenFile, ChunksizeAdjuster
def requires_multipart_upload(self, transfer_future, config):
    if transfer_future.meta.size is not None:
        return transfer_future.meta.size >= config.multipart_threshold
    fileobj = transfer_future.meta.call_args.fileobj
    threshold = config.multipart_threshold
    self._initial_data = self._read(fileobj, threshold, False)
    if len(self._initial_data) < threshold:
        return False
    else:
        return True