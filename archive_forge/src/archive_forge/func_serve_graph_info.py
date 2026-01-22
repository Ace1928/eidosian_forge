import threading
from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.plugins import base_plugin
from tensorboard.plugins.debugger_v2 import debug_data_provider
from tensorboard.backend import http_util
@wrappers.Request.application
def serve_graph_info(self, request):
    """Serve basic information about a TensorFlow graph.

        The request specifies the debugger-generated ID of the graph being
        queried.

        The response contains a JSON object with the following fields:
          - graph_id: The debugger-generated ID (echoing the request).
          - name: The name of the graph (if any). For TensorFlow 2.x
            Function Graphs (FuncGraphs), this is typically the name of
            the underlying Python function, optionally prefixed with
            TensorFlow-generated prefixed such as "__inference_".
            Some graphs (e.g., certain outermost graphs) may have no names,
            in which case this field is `null`.
          - outer_graph_id: Outer graph ID (if any). For an outermost graph
            without an outer graph context, this field is `null`.
          - inner_graph_ids: Debugger-generated IDs of all the graphs
            nested inside this graph. For a graph without any graphs nested
            inside, this field is an empty array.
        """
    experiment = plugin_util.experiment_id(request.environ)
    run = request.args.get('run')
    if run is None:
        return _missing_run_error_response(request)
    graph_id = request.args.get('graph_id')
    run_tag_filter = debug_data_provider.graph_info_run_tag_filter(run, graph_id)
    blob_sequences = self._data_provider.read_blob_sequences(experiment_id=experiment, plugin_name=self.plugin_name, run_tag_filter=run_tag_filter)
    tag = next(iter(run_tag_filter.tags))
    try:
        return http_util.Respond(request, self._data_provider.read_blob(blob_key=blob_sequences[run][tag][0].blob_key), 'application/json')
    except errors.NotFoundError as e:
        return _error_response(request, str(e))