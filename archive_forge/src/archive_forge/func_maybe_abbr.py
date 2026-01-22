import sys
from operator import itemgetter
import click
from celery.bin.base import CeleryCommand, handle_preload_options
from celery.utils.graph import DependencyGraph, GraphFormatter
def maybe_abbr(l, name, max=Wmax):
    size = len(l)
    abbr = max and size > max
    if 'enumerate' in args:
        l = [f'{name}{subscript(i + 1)}' for i, obj in enumerate(l)]
    if abbr:
        l = l[0:max - 1] + [l[size - 1]]
        l[max - 2] = '{}⎨…{}⎬'.format(name[0], subscript(size - (max - 1)))
    return l