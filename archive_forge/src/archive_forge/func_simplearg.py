import sys
from operator import itemgetter
import click
from celery.bin.base import CeleryCommand, handle_preload_options
from celery.utils.graph import DependencyGraph, GraphFormatter
def simplearg(arg):
    return maybe_list(itemgetter(0, 2)(arg.partition(':')))