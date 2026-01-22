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
def secho(self, message=None, **kwargs):
    if self.no_color:
        kwargs['color'] = False
        click.echo(message, **kwargs)
    else:
        click.secho(message, **kwargs)