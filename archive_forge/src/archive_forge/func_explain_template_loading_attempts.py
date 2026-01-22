from __future__ import annotations
import typing as t
from jinja2.loaders import BaseLoader
from werkzeug.routing import RequestRedirect
from .blueprints import Blueprint
from .globals import request_ctx
from .sansio.app import App
def explain_template_loading_attempts(app: App, template: str, attempts: list[tuple[BaseLoader, Scaffold, tuple[str, str | None, t.Callable[[], bool] | None] | None]]) -> None:
    """This should help developers understand what failed"""
    info = [f'Locating template {template!r}:']
    total_found = 0
    blueprint = None
    if request_ctx and request_ctx.request.blueprint is not None:
        blueprint = request_ctx.request.blueprint
    for idx, (loader, srcobj, triple) in enumerate(attempts):
        if isinstance(srcobj, App):
            src_info = f'application {srcobj.import_name!r}'
        elif isinstance(srcobj, Blueprint):
            src_info = f'blueprint {srcobj.name!r} ({srcobj.import_name})'
        else:
            src_info = repr(srcobj)
        info.append(f'{idx + 1:5}: trying loader of {src_info}')
        for line in _dump_loader_info(loader):
            info.append(f'       {line}')
        if triple is None:
            detail = 'no match'
        else:
            detail = f'found ({triple[1] or '<string>'!r})'
            total_found += 1
        info.append(f'       -> {detail}')
    seems_fishy = False
    if total_found == 0:
        info.append('Error: the template could not be found.')
        seems_fishy = True
    elif total_found > 1:
        info.append('Warning: multiple loaders returned a match for the template.')
        seems_fishy = True
    if blueprint is not None and seems_fishy:
        info.append(f'  The template was looked up from an endpoint that belongs to the blueprint {blueprint!r}.')
        info.append('  Maybe you did not place a template in the right folder?')
        info.append('  See https://flask.palletsprojects.com/blueprints/#templates')
    app.logger.info('\n'.join(info))