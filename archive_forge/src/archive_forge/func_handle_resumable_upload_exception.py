import errno
import os
import random
import re
import socket
import time
from hashlib import md5
import six.moves.http_client as httplib
from six.moves import urllib as urlparse
from boto import config, UserAgent
from boto.connection import AWSAuthConnection
from boto.exception import InvalidUriError
from boto.exception import ResumableTransferDisposition
from boto.exception import ResumableUploadException
from boto.s3.keyfile import KeyFile
def handle_resumable_upload_exception(self, e, debug):
    if e.disposition == ResumableTransferDisposition.ABORT_CUR_PROCESS:
        if debug >= 1:
            print('Caught non-retryable ResumableUploadException (%s); aborting but retaining tracker file' % e.message)
        raise
    elif e.disposition == ResumableTransferDisposition.ABORT:
        if debug >= 1:
            print('Caught non-retryable ResumableUploadException (%s); aborting and removing tracker file' % e.message)
        self._remove_tracker_file()
        raise
    elif debug >= 1:
        print('Caught ResumableUploadException (%s) - will retry' % e.message)