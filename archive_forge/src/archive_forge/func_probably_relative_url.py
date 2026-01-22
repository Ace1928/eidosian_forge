from __future__ import annotations
import logging # isort:skip
import importlib.util
from os.path import isfile
from types import ModuleType
from typing import (
from tornado.httputil import HTTPServerRequest
from tornado.web import RequestHandler
from ..util.serialization import make_globally_unique_id
def probably_relative_url(url: str) -> bool:
    """ Return True if a URL is not one of the common absolute URL formats.

    Arguments:
        url (str): a URL string

    Returns
        bool

    """
    return not url.startswith(('http://', 'https://', '//'))