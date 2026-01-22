from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.spanner import samples
from googlecloudsdk.core import execution_utils
def run_workload(appname, port=None, capture_logs=False):
    """Run the workload generator executable for the given sample app.

  Args:
    appname: str, Name of the sample app.
    port: int, Port to run the service on.
    capture_logs: bool, Whether to save logs to disk or print to stdout.

  Returns:
    subprocess.Popen or execution_utils.SubprocessTimeoutWrapper, The running
      subprocess.
  """
    proc_args = ['java', '-jar', _get_popen_jar(appname)]
    if port is not None:
        proc_args.append('--port={}'.format(port))
    capture_logs_fn = os.path.join(samples.SAMPLES_LOG_PATH, '{}-workload.log'.format(appname)) if capture_logs else None
    return samples.run_proc(proc_args, capture_logs_fn)