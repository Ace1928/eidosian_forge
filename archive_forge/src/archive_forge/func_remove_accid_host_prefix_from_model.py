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
def remove_accid_host_prefix_from_model(params, model, context, **kwargs):
    """Removes the `{AccountId}.` prefix from the operation model.

    This change is applied to the operation model during the first time the
    operation is invoked and then stays in effect for the lifetime of the
    client object.

    When the ruleset based endpoint resolver is in effect, both the endpoint
    ruleset AND the service model place the {AccountId}. prefix in the URL.
    The result is an invalid endpoint. This handler modifies the operation
    model to remove the `endpoint.hostPrefix` field while leaving the
    `RequiresAccountId` static context parameter in place.
    """
    has_ctx_param = any((ctx_param.name == 'RequiresAccountId' and ctx_param.value is True for ctx_param in model.static_context_parameters))
    if model.endpoint is not None and model.endpoint.get('hostPrefix') == '{AccountId}.' and has_ctx_param:
        del model.endpoint['hostPrefix']