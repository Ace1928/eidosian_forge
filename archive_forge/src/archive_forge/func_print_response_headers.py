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
def print_response_headers(http_version: bytes, status: int, reason_phrase: bytes | None, headers: list[tuple[bytes, bytes]]) -> None:
    console = rich.console.Console()
    http_text = format_response_headers(http_version, status, reason_phrase, headers)
    syntax = rich.syntax.Syntax(http_text, 'http', theme='ansi_dark', word_wrap=True)
    console.print(syntax)
    syntax = rich.syntax.Syntax('', 'http', theme='ansi_dark', word_wrap=True)
    console.print(syntax)