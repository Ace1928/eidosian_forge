import json
import numbers
from collections import OrderedDict
from functools import update_wrapper
from pprint import pformat
from typing import Any
import click
from click import Context, ParamType
from kombu.utils.objects import cached_property
from celery._state import get_current_app
from celery.signals import user_preload_options
from celery.utils import text
from celery.utils.log import mlevel
from celery.utils.time import maybe_iso8601
def handle_preload_options(f):
    """Extract preload options and return a wrapped callable."""

    def caller(ctx, *args, **kwargs):
        app = ctx.obj.app
        preload_options = [o.name for o in app.user_options.get('preload', [])]
        if preload_options:
            user_options = {preload_option: kwargs[preload_option] for preload_option in preload_options}
            user_preload_options.send(sender=f, app=app, options=user_options)
        return f(ctx, *args, **kwargs)
    return update_wrapper(caller, f)