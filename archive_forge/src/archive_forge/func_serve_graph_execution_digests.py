import threading
from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.plugins import base_plugin
from tensorboard.plugins.debugger_v2 import debug_data_provider
from tensorboard.backend import http_util
@wrappers.Request.application
def serve_graph_execution_digests(self, request):
    """Serve digests of intra-graph execution events.

        As the names imply, this route differs from `serve_execution_digests()`
        in that it is for intra-graph execution, while `serve_execution_digests()`
        is for top-level (eager) execution.
        """
    experiment = plugin_util.experiment_id(request.environ)
    run = request.args.get('run')
    if run is None:
        return _missing_run_error_response(request)
    begin = int(request.args.get('begin', '0'))
    end = int(request.args.get('end', '-1'))
    run_tag_filter = debug_data_provider.graph_execution_digest_run_tag_filter(run, begin, end)
    blob_sequences = self._data_provider.read_blob_sequences(experiment_id=experiment, plugin_name=self.plugin_name, run_tag_filter=run_tag_filter)
    tag = next(iter(run_tag_filter.tags))
    try:
        return http_util.Respond(request, self._data_provider.read_blob(blob_key=blob_sequences[run][tag][0].blob_key), 'application/json')
    except errors.InvalidArgumentError as e:
        return _error_response(request, str(e))