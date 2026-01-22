import errno
import inspect
import json
import logging
import os
import re
import select
import socket
import sys
import time
import traceback
import uuid
from pdb import Pdb
from typing import Callable
import setproctitle
import ray
from ray._private import ray_constants
from ray.experimental.internal_kv import _internal_kv_del, _internal_kv_put
from ray.util.annotations import DeveloperAPI
def do_remote(self, arg):
    """remote
        Skip into the next remote call.
        """
    ray._private.worker.global_worker.debugger_breakpoint = self._breakpoint_uuid
    data = json.dumps({'job_id': ray.get_runtime_context().get_job_id()})
    _internal_kv_put('RAY_PDB_CONTINUE_{}'.format(self._breakpoint_uuid), data, namespace=ray_constants.KV_NAMESPACE_PDB)
    self.__restore()
    self.handle.connection.close()
    return Pdb.do_continue(self, arg)