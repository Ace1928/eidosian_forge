import collections
import logging
import os
import re
import sys
import time
import warnings
from contextlib import contextmanager
from functools import wraps
from io import StringIO
from itertools import chain
from types import SimpleNamespace
from unittest import TestCase, skipIf, skipUnless
from xml.dom.minidom import Node, parseString
from asgiref.sync import iscoroutinefunction
from django.apps import apps
from django.apps.registry import Apps
from django.conf import UserSettingsHolder, settings
from django.core import mail
from django.core.exceptions import ImproperlyConfigured
from django.core.signals import request_started, setting_changed
from django.db import DEFAULT_DB_ALIAS, connections, reset_queries
from django.db.models.options import Options
from django.template import Template
from django.test.signals import template_rendered
from django.urls import get_script_prefix, set_script_prefix
from django.utils.translation import deactivate
def setup_test_environment(debug=None):
    """
    Perform global pre-test setup, such as installing the instrumented template
    renderer and setting the email backend to the locmem email backend.
    """
    if hasattr(_TestState, 'saved_data'):
        raise RuntimeError("setup_test_environment() was already called and can't be called again without first calling teardown_test_environment().")
    if debug is None:
        debug = settings.DEBUG
    saved_data = SimpleNamespace()
    _TestState.saved_data = saved_data
    saved_data.allowed_hosts = settings.ALLOWED_HOSTS
    settings.ALLOWED_HOSTS = [*settings.ALLOWED_HOSTS, 'testserver']
    saved_data.debug = settings.DEBUG
    settings.DEBUG = debug
    saved_data.email_backend = settings.EMAIL_BACKEND
    settings.EMAIL_BACKEND = 'django.core.mail.backends.locmem.EmailBackend'
    saved_data.template_render = Template._render
    Template._render = instrumented_test_render
    mail.outbox = []
    deactivate()