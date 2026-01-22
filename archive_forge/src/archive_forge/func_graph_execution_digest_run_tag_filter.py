import json
from tensorboard.data import provider
from tensorboard.plugins.debugger_v2 import debug_data_multiplexer
def graph_execution_digest_run_tag_filter(run, begin, end, trace_id=None):
    """Create a RunTagFilter for GraphExecutionTraceDigests.

    This differs from `graph_execution_data_run_tag_filter()` in that it is for
    the small-size digest objects for intra-graph execution debug events, instead
    of the full-size data objects.

    Args:
      run: tfdbg2 run name.
      begin: Beginning index of GraphExecutionTraceDigests.
      end: Ending index of GraphExecutionTraceDigests.

    Returns:
      `RunTagFilter` for the run and range of GraphExecutionTraceDigests.
    """
    if trace_id is not None:
        raise NotImplementedError('trace_id support for graph_execution_digest_run_tag_filter() is not implemented yet.')
    return provider.RunTagFilter(runs=[run], tags=['%s_%d_%d' % (GRAPH_EXECUTION_DIGESTS_BLOB_TAG_PREFIX, begin, end)])