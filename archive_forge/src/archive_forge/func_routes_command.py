from __future__ import annotations
import ast
import collections.abc as cabc
import importlib.metadata
import inspect
import os
import platform
import re
import sys
import traceback
import typing as t
from functools import update_wrapper
from operator import itemgetter
from types import ModuleType
import click
from click.core import ParameterSource
from werkzeug import run_simple
from werkzeug.serving import is_running_from_reloader
from werkzeug.utils import import_string
from .globals import current_app
from .helpers import get_debug_flag
from .helpers import get_load_dotenv
@click.command('routes', short_help='Show the routes for the app.')
@click.option('--sort', '-s', type=click.Choice(('endpoint', 'methods', 'domain', 'rule', 'match')), default='endpoint', help="Method to sort routes by. 'match' is the order that Flask will match routes when dispatching a request.")
@click.option('--all-methods', is_flag=True, help='Show HEAD and OPTIONS methods.')
@with_appcontext
def routes_command(sort: str, all_methods: bool) -> None:
    """Show all registered routes with endpoints and methods."""
    rules = list(current_app.url_map.iter_rules())
    if not rules:
        click.echo('No routes were registered.')
        return
    ignored_methods = set() if all_methods else {'HEAD', 'OPTIONS'}
    host_matching = current_app.url_map.host_matching
    has_domain = any((rule.host if host_matching else rule.subdomain for rule in rules))
    rows = []
    for rule in rules:
        row = [rule.endpoint, ', '.join(sorted((rule.methods or set()) - ignored_methods))]
        if has_domain:
            row.append((rule.host if host_matching else rule.subdomain) or '')
        row.append(rule.rule)
        rows.append(row)
    headers = ['Endpoint', 'Methods']
    sorts = ['endpoint', 'methods']
    if has_domain:
        headers.append('Host' if host_matching else 'Subdomain')
        sorts.append('domain')
    headers.append('Rule')
    sorts.append('rule')
    try:
        rows.sort(key=itemgetter(sorts.index(sort)))
    except ValueError:
        pass
    rows.insert(0, headers)
    widths = [max((len(row[i]) for row in rows)) for i in range(len(headers))]
    rows.insert(1, ['-' * w for w in widths])
    template = '  '.join((f'{{{i}:<{w}}}' for i, w in enumerate(widths)))
    for row in rows:
        click.echo(template.format(*row))