import logging
import os
import re
def invoke_initializers(session):
    """Invoke all initializers for a session.

    :type session: botocore.session.Session
    :param session: The session to initialize.

    """
    for initializer in _INITIALIZERS:
        initializer(session)