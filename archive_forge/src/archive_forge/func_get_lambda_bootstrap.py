import sys
from copy import deepcopy
from datetime import timedelta
from os import environ
from sentry_sdk.api import continue_trace
from sentry_sdk.consts import OP
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.tracing import TRANSACTION_SOURCE_COMPONENT
from sentry_sdk.utils import (
from sentry_sdk.integrations import Integration
from sentry_sdk.integrations._wsgi_common import _filter_headers
from sentry_sdk._compat import datetime_utcnow, reraise
from sentry_sdk._types import TYPE_CHECKING
def get_lambda_bootstrap():
    if 'bootstrap' in sys.modules:
        return sys.modules['bootstrap']
    elif '__main__' in sys.modules:
        module = sys.modules['__main__']
        if hasattr(module, 'awslambdaricmain') and hasattr(module.awslambdaricmain, 'bootstrap'):
            return module.awslambdaricmain.bootstrap
        elif hasattr(module, 'bootstrap'):
            return module.bootstrap
        return module
    else:
        return None