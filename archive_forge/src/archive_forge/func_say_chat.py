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
def say_chat(self, direction, title, body='', show_body=False):
    if direction == '<-' and self.quiet:
        return
    dirstr = not self.quiet and f'{self.style(direction, fg='white', bold=True)} ' or ''
    self.echo(f'{dirstr} {title}')
    if body and show_body:
        self.echo(body)