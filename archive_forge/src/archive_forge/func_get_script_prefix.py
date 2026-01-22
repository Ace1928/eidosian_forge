import asyncio
import logging
import sys
import tempfile
import traceback
from contextlib import aclosing
from asgiref.sync import ThreadSensitiveContext, sync_to_async
from django.conf import settings
from django.core import signals
from django.core.exceptions import RequestAborted, RequestDataTooBig
from django.core.handlers import base
from django.http import (
from django.urls import set_script_prefix
from django.utils.functional import cached_property
def get_script_prefix(scope):
    """
    Return the script prefix to use from either the scope or a setting.
    """
    if settings.FORCE_SCRIPT_NAME:
        return settings.FORCE_SCRIPT_NAME
    return scope.get('root_path', '') or ''