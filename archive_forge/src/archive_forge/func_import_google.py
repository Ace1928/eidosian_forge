from __future__ import annotations
import logging
import os
from typing import TYPE_CHECKING, List, Optional, Tuple
def import_google() -> Tuple[Request, Credentials]:
    """Import google libraries.

    Returns:
        Tuple[Request, Credentials]: Request and Credentials classes.
    """
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
    except ImportError:
        raise ImportError('You need to install google-auth-httplib2 to use this toolkit. Try running pip install --upgrade google-auth-httplib2')
    return (Request, Credentials)