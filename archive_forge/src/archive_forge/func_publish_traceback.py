import signal
import sys
import traceback
from tensorflow.python.debug.lib import common
from tensorflow.python.debug.wrappers import framework
def publish_traceback(debug_server_urls, graph, feed_dict, fetches, old_graph_version):
    """Publish traceback and source code if graph version is new.

  `graph.version` is compared with `old_graph_version`. If the former is higher
  (i.e., newer), the graph traceback and the associated source code is sent to
  the debug server at the specified gRPC URLs.

  Args:
    debug_server_urls: A single gRPC debug server URL as a `str` or a `list` of
      debug server URLs.
    graph: A Python `tf.Graph` object.
    feed_dict: Feed dictionary given to the `Session.run()` call.
    fetches: Fetches from the `Session.run()` call.
    old_graph_version: Old graph version to compare to.

  Returns:
    If `graph.version > old_graph_version`, the new graph version as an `int`.
    Else, the `old_graph_version` is returned.
  """
    from tensorflow.python.debug.lib import source_remote
    if graph.version > old_graph_version:
        run_key = common.get_run_key(feed_dict, fetches)
        source_remote.send_graph_tracebacks(debug_server_urls, run_key, traceback.extract_stack(), graph, send_source=True)
        return graph.version
    else:
        return old_graph_version