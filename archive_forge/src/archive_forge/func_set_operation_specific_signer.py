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
def set_operation_specific_signer(context, signing_name, **kwargs):
    """Choose the operation-specific signer.

    Individual operations may have a different auth type than the service as a
    whole. This will most often manifest as operations that should not be
    authenticated at all, but can include other auth modes such as sigv4
    without body signing.
    """
    auth_type = context.get('auth_type')
    if not auth_type:
        return
    if auth_type == 'none':
        return botocore.UNSIGNED
    if auth_type == 'bearer':
        return 'bearer'
    if auth_type.startswith('v4'):
        if auth_type == 'v4-s3express':
            return auth_type
        if auth_type == 'v4a':
            signing = {'region': '*', 'signing_name': signing_name}
            if 'signing' in context:
                context['signing'].update(signing)
            else:
                context['signing'] = signing
            signature_version = 'v4a'
        else:
            signature_version = 'v4'
        if auth_type == 'v4-unsigned-body':
            context['payload_signing_enabled'] = False
        if signing_name in S3_SIGNING_NAMES:
            signature_version = f's3{signature_version}'
        return signature_version