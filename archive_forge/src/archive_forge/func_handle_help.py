from __future__ import annotations
import functools
import json
import sys
import typing
import click
import httpcore
import pygments.lexers
import pygments.util
import rich.console
import rich.markup
import rich.progress
import rich.syntax
import rich.table
from ._client import Client
from ._exceptions import RequestError
from ._models import Response
from ._status_codes import codes
def handle_help(ctx: click.Context, param: click.Option | click.Parameter, value: typing.Any) -> None:
    if not value or ctx.resilient_parsing:
        return
    print_help()
    ctx.exit()