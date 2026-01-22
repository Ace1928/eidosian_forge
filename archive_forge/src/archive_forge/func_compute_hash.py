import base64
import binascii
import os
import re
from boto.compat import StringIO, six
from boto.exception import BotoClientError
from boto.s3.key import Key as S3Key
from boto.s3.keyfile import KeyFile
from boto.utils import compute_hash, get_utf8able_str
def compute_hash(self, fp, algorithm, size=None):
    """
        :type fp: file
        :param fp: File pointer to the file to hash. The file
            pointer will be reset to the same position before the
            method returns.

        :type algorithm: zero-argument constructor for hash objects that
            implements update() and digest() (e.g. hashlib.md5)

        :type size: int
        :param size: (optional) The Maximum number of bytes to read
            from the file pointer (fp). This is useful when uploading
            a file in multiple parts where the file is being split
            in place into different parts. Less bytes may be available.
        """
    hex_digest, b64_digest, data_size = compute_hash(fp, size=size, hash_algorithm=algorithm)
    self.size = data_size
    return (hex_digest, b64_digest)