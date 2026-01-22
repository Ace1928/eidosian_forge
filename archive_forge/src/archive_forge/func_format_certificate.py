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
def format_certificate(cert: _PeerCertRetDictType) -> str:
    lines = []
    for key, value in cert.items():
        if isinstance(value, (list, tuple)):
            lines.append(f'*   {key}:')
            for item in value:
                if key in ('subject', 'issuer'):
                    for sub_item in item:
                        lines.append(f'*     {sub_item[0]}: {sub_item[1]!r}')
                elif isinstance(item, tuple) and len(item) == 2:
                    lines.append(f'*     {item[0]}: {item[1]!r}')
                else:
                    lines.append(f'*     {item!r}')
        else:
            lines.append(f'*   {key}: {value!r}')
    return '\n'.join(lines)