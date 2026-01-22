from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import itertools
import json
import threading
import time
import boto
from boto.gs.acl import CannedACLStrings
from gslib.storage_url import IsFileUrlString
from gslib.storage_url import StorageUrlFromString
from gslib.storage_url import StripOneSlash
from gslib.utils.boto_util import GetTabCompletionCacheFilename
from gslib.utils.boto_util import GetTabCompletionLogFilename
from gslib.wildcard_iterator import CreateWildcardIterator
Perform a remote listing request for the given wildcard URL.

    Args:
      wildcard_url: The wildcard URL to list.
      timeout: Time limit for the request.
    Returns:
      Cloud resources matching the given wildcard URL.
    Raises:
      TimeoutError: If the listing does not finish within the timeout.
    