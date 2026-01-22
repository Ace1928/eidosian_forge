import os
import sys
import atexit
import signal
import logging
from pprint import pformat
from logging import NullHandler
import click
from tornado.options import options
from tornado.options import parse_command_line, parse_config_file
from tornado.log import enable_pretty_logging
from celery.bin.base import CeleryCommand
from .app import Flower
from .urls import settings
from .utils import abs_path, prepend_url, strtobool
from .options import DEFAULT_CONFIG_FILE, default_options
from .views.auth import validate_auth_option
@click.command(cls=CeleryCommand, context_settings={'ignore_unknown_options': True})
@click.argument('tornado_argv', nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def flower(ctx, tornado_argv):
    """Web based tool for monitoring and administrating Celery clusters."""
    warn_about_celery_args_used_in_flower_command(ctx, tornado_argv)
    apply_env_options()
    apply_options(sys.argv[0], tornado_argv)
    extract_settings()
    setup_logging()
    app = ctx.obj.app
    flower_app = Flower(capp=app, options=options, **settings)
    atexit.register(flower_app.stop)
    signal.signal(signal.SIGTERM, sigterm_handler)
    if not ctx.obj.quiet:
        print_banner(app, 'ssl_options' in settings)
    try:
        flower_app.start()
    except (KeyboardInterrupt, SystemExit):
        pass