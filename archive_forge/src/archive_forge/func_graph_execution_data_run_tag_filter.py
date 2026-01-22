import json
from tensorboard.data import provider
from tensorboard.plugins.debugger_v2 import debug_data_multiplexer
def graph_execution_data_run_tag_filter(run, begin, end, trace_id=None):
    """Create a RunTagFilter for GraphExecutionTrace.

    This method differs from `graph_execution_digest_run_tag_filter()` in that
    it is for full-sized data objects for intra-graph execution events.

    Args:
      run: tfdbg2 run name.
      begin: Beginning index of GraphExecutionTrace.
      end: Ending index of GraphExecutionTrace.

    Returns:
      `RunTagFilter` for the run and range of GraphExecutionTrace.
    """
    if trace_id is not None:
        raise NotImplementedError('trace_id support for graph_execution_data_run_tag_filter() is not implemented yet.')
    return provider.RunTagFilter(runs=[run], tags=['%s_%d_%d' % (GRAPH_EXECUTION_DATA_BLOB_TAG_PREFIX, begin, end)])