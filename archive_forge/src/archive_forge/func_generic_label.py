import sys
from operator import itemgetter
import click
from celery.bin.base import CeleryCommand, handle_preload_options
from celery.utils.graph import DependencyGraph, GraphFormatter
def generic_label(node):
    return '{} ({}://)'.format(type(node).__name__, node._label.split('://')[0])