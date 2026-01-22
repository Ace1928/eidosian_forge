import errno
import os
import re
import socket
import time
import six.moves.http_client as httplib
import boto
from boto import config, storage_uri_for_key
from boto.connection import AWSAuthConnection
from boto.exception import ResumableDownloadException
from boto.exception import ResumableTransferDisposition
from boto.s3.keyfile import KeyFile
from boto.gs.key import Key as GSKey

        Retrieves a file from a Key
        :type key: :class:`boto.s3.key.Key` or subclass
        :param key: The Key object from which upload is to be downloaded
        
        :type fp: file
        :param fp: File pointer into which data should be downloaded
        
        :type headers: string
        :param: headers to send when retrieving the files
        
        :type cb: function
        :param cb: (optional) a callback function that will be called to report
             progress on the download.  The callback should accept two integer
             parameters, the first representing the number of bytes that have
             been successfully transmitted from the storage service and
             the second representing the total number of bytes that need
             to be transmitted.
        
        :type num_cb: int
        :param num_cb: (optional) If a callback is specified with the cb
             parameter this parameter determines the granularity of the callback
             by defining the maximum number of times the callback will be
             called during the file transfer.
             
        :type torrent: bool
        :param torrent: Flag for whether to get a torrent for the file

        :type version_id: string
        :param version_id: The version ID (optional)

        :type hash_algs: dictionary
        :param hash_algs: (optional) Dictionary of hash algorithms and
            corresponding hashing class that implements update() and digest().
            Defaults to {'md5': hashlib/md5.md5}.

        Raises ResumableDownloadException if a problem occurs during
            the transfer.
        