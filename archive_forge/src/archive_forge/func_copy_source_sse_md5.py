import base64
import copy
import logging
import os
import re
import uuid
import warnings
from io import BytesIO
import botocore
import botocore.auth
from botocore import utils
from botocore.compat import (
from botocore.docs.utils import (
from botocore.endpoint_provider import VALID_HOST_LABEL_RE
from botocore.exceptions import (
from botocore.regions import EndpointResolverBuiltins
from botocore.signers import (
from botocore.utils import (
from botocore import retryhandler  # noqa
from botocore import translate  # noqa
from botocore.compat import MD5_AVAILABLE  # noqa
from botocore.exceptions import MissingServiceIdError  # noqa
from botocore.utils import hyphenize_service_id  # noqa
from botocore.utils import is_global_accesspoint  # noqa
from botocore.utils import SERVICE_NAME_ALIASES  # noqa
def copy_source_sse_md5(params, **kwargs):
    """
    S3 server-side encryption requires the encryption key to be sent to the
    server base64 encoded, as well as a base64-encoded MD5 hash of the
    encryption key. This handler does both if the MD5 has not been set by
    the caller specifically if the parameter is for the copy-source sse-c key.
    """
    _sse_md5(params, 'CopySourceSSECustomer')