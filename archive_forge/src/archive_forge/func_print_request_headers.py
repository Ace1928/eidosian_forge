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
def print_request_headers(request: httpcore.Request, http2: bool=False) -> None:
    console = rich.console.Console()
    http_text = format_request_headers(request, http2=http2)
    syntax = rich.syntax.Syntax(http_text, 'http', theme='ansi_dark', word_wrap=True)
    console.print(syntax)
    syntax = rich.syntax.Syntax('', 'http', theme='ansi_dark', word_wrap=True)
    console.print(syntax)