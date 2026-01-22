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
def remove_arn_from_signing_path(request, **kwargs):
    auth_path = request.auth_path
    if isinstance(auth_path, str) and auth_path.startswith('/arn%3A'):
        auth_path_parts = auth_path.split('/')
        if len(auth_path_parts) > 1 and ArnParser.is_arn(unquote(auth_path_parts[1])):
            request.auth_path = '/'.join(['', *auth_path_parts[2:]])