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
def inject_presigned_url_ec2(params, request_signer, model, **kwargs):
    if 'PresignedUrl' in params['body']:
        return
    src, dest = _get_presigned_url_source_and_destination_regions(request_signer, params['body'])
    url = _get_cross_region_presigned_url(request_signer, params, model, src, dest)
    params['body']['PresignedUrl'] = url
    params['body']['DestinationRegion'] = dest