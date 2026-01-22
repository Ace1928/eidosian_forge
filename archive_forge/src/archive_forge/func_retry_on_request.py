import functools
import logging
import random
import threading
import time
from oslo_utils import excutils
from oslo_utils import importutils
from oslo_utils import reflection
from oslo_db import exception
from oslo_db import options
def retry_on_request(f):
    """Retry a DB API call if RetryRequest exception was received.

    wrap_db_entry will be applied to all db.api functions marked with this
    decorator.
    """
    f.__dict__['enable_retry_on_request'] = True
    return f