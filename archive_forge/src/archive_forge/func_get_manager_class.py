import functools
import logging
from multiprocessing import managers
import os
import shutil
import signal
import stat
import sys
import tempfile
import threading
import time
from oslo_rootwrap import cmd
from oslo_rootwrap import jsonrpc
from oslo_rootwrap import subprocess
from oslo_rootwrap import wrapper
def get_manager_class(config=None, filters=None):

    class RootwrapManager(managers.BaseManager):

        def __init__(self, address=None, authkey=None):
            super(RootwrapManager, self).__init__(address, authkey, serializer='jsonrpc')
    if config is not None:
        partial_class = functools.partial(RootwrapClass, config, filters)
        RootwrapManager.register('rootwrap', partial_class)
    else:
        RootwrapManager.register('rootwrap')
    return RootwrapManager