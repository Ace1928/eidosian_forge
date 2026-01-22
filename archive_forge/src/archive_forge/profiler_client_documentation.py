from tensorflow.python.profiler.internal import _pywrap_profiler
from tensorflow.python.util.deprecation import deprecated
Sends grpc requests to profiler server to perform on-demand monitoring.

  This method will block caller thread until receives monitoring result.

  Args:
    service_addr: Address of profiler service e.g. localhost:6009.
    duration_ms: Duration of tracing or monitoring in ms.
    monitoring_level: Choose a monitoring level between 1 and 2 to monitor your
      job. Level 2 is more verbose than level 1 and shows more metrics.
    display_timestamp: Set to true to display timestamp in monitoring result.

  Returns:
    A string of monitoring output.
  