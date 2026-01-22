import contextlib
import copy
import json
import os
import subprocess
import sys
import threading
import unittest
import six
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.distribute import distribute_coordinator as dc
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute.cluster_resolver import cluster_resolver as cluster_resolver_lib
from tensorflow.python.distribute.cluster_resolver import tfconfig_cluster_resolver
from tensorflow.python.eager import context
from tensorflow.python.eager import remote
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import coordinator
from tensorflow.python.training import server_lib
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
@deprecation.deprecated(None, '`stream_stderr` is deprecated; any new test requiring multiple processes should use `multi_process_runner` for better support of log printing, streaming, and more functionality.')
def stream_stderr(self, processes, print_only_first=False):
    """Consume stderr of all processes and print to stdout.

    To reduce the amount of logging, caller can set print_only_first to True.
    In that case, this function only prints stderr from the first process of
    each type.

    Args:
      processes: A dictionary from process type string -> list of processes.
      print_only_first: If true, only print output from first process of each
        type.
    """

    def _stream_stderr_single_process(process, type_string, index, print_to_stdout):
        """Consume a single process's stderr and optionally print to stdout."""
        while True:
            output = process.stderr.readline()
            if not output and process.poll() is not None:
                break
            if output and print_to_stdout:
                print('{}{} {}'.format(type_string, index, output.strip()))
                sys.stdout.flush()
    stream_threads = []
    for process_type, process_list in six.iteritems(processes):
        for i in range(len(process_list)):
            print_to_stdout = not print_only_first or i == 0
            thread = threading.Thread(target=_stream_stderr_single_process, args=(process_list[i], process_type, i, print_to_stdout))
            thread.start()
            stream_threads.append(thread)
    for thread in stream_threads:
        thread.join()