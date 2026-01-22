import os
import signal
import sys
from functools import wraps
import click
from kombu.utils.objects import cached_property
from celery import VERSION_BANNER
from celery.apps.multi import Cluster, MultiParser, NamespacedOptionParser
from celery.bin.base import CeleryCommand, handle_preload_options
from celery.platforms import EX_FAILURE, EX_OK, signals
from celery.utils import term
from celery.utils.text import pluralize
def using_cluster_and_sig(fun):

    @wraps(fun)
    def _inner(self, *argv, **kwargs):
        p, cluster = self._cluster_from_argv(argv)
        sig = self._find_sig_argument(p)
        return fun(self, cluster, sig, **kwargs)
    return _inner