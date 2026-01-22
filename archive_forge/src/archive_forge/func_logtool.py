import re
from collections import Counter
from fileinput import FileInput
import click
from celery.bin.base import CeleryCommand, handle_preload_options
@click.group()
@click.pass_context
@handle_preload_options
def logtool(ctx):
    """The ``celery logtool`` command."""