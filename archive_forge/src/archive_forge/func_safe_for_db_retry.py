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
def safe_for_db_retry(f):
    """Indicate api method as safe for re-connection to database.

    Database connection retries will be enabled for the decorated api method.
    Database connection failure can have many causes, which can be temporary.
    In such cases retry may increase the likelihood of connection.

    Usage::

        @safe_for_db_retry
        def api_method(self):
            self.engine.connect()


    :param f: database api method.
    :type f: function.
    """
    f.__dict__['enable_retry_on_disconnect'] = True
    return f