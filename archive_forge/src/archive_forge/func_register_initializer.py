import logging
import os
import re
def register_initializer(callback):
    """Register an initializer function for session creation.

    This initializer function will be invoked whenever a new
    `botocore.session.Session` is instantiated.

    :type callback: callable
    :param callback: A callable that accepts a single argument
        of type `botocore.session.Session`.

    """
    _INITIALIZERS.append(callback)