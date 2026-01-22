import json
from tensorboard.data import provider
from tensorboard.plugins.debugger_v2 import debug_data_multiplexer
def read_blob_sequences(self, ctx=None, *, experiment_id, plugin_name, downsample=None, run_tag_filter=None):
    del experiment_id, downsample
    if plugin_name != PLUGIN_NAME:
        raise ValueError('Unsupported plugin_name: %s' % plugin_name)
    if run_tag_filter.runs is None:
        raise ValueError('run_tag_filter.runs is expected to be specified, but is not.')
    if run_tag_filter.tags is None:
        raise ValueError('run_tag_filter.tags is expected to be specified, but is not.')
    output = dict()
    existing_runs = self._multiplexer.Runs()
    for run in run_tag_filter.runs:
        if run not in existing_runs:
            continue
        output[run] = dict()
        for tag in run_tag_filter.tags:
            if tag.startswith((ALERTS_BLOB_TAG_PREFIX, EXECUTION_DIGESTS_BLOB_TAG_PREFIX, EXECUTION_DATA_BLOB_TAG_PREFIX, GRAPH_EXECUTION_DIGESTS_BLOB_TAG_PREFIX, GRAPH_EXECUTION_DATA_BLOB_TAG_PREFIX, GRAPH_INFO_BLOB_TAG_PREFIX, GRAPH_OP_INFO_BLOB_TAG_PREFIX, SOURCE_FILE_BLOB_TAG_PREFIX, STACK_FRAMES_BLOB_TAG_PREFIX)) or tag in (SOURCE_FILE_LIST_BLOB_TAG,):
                output[run][tag] = [provider.BlobReference(blob_key='%s.%s' % (tag, run))]
    return output