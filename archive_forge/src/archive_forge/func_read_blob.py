import json
from tensorboard.data import provider
from tensorboard.plugins.debugger_v2 import debug_data_multiplexer
def read_blob(self, ctx=None, *, blob_key):
    if blob_key.startswith(ALERTS_BLOB_TAG_PREFIX):
        run, begin, end, alert_type = _parse_alerts_blob_key(blob_key)
        return json.dumps(self._multiplexer.Alerts(run, begin, end, alert_type_filter=alert_type))
    elif blob_key.startswith(EXECUTION_DIGESTS_BLOB_TAG_PREFIX):
        run, begin, end = _parse_execution_digest_blob_key(blob_key)
        return json.dumps(self._multiplexer.ExecutionDigests(run, begin, end))
    elif blob_key.startswith(EXECUTION_DATA_BLOB_TAG_PREFIX):
        run, begin, end = _parse_execution_data_blob_key(blob_key)
        return json.dumps(self._multiplexer.ExecutionData(run, begin, end))
    elif blob_key.startswith(GRAPH_EXECUTION_DIGESTS_BLOB_TAG_PREFIX):
        run, begin, end = _parse_graph_execution_digest_blob_key(blob_key)
        return json.dumps(self._multiplexer.GraphExecutionDigests(run, begin, end))
    elif blob_key.startswith(GRAPH_EXECUTION_DATA_BLOB_TAG_PREFIX):
        run, begin, end = _parse_graph_execution_data_blob_key(blob_key)
        return json.dumps(self._multiplexer.GraphExecutionData(run, begin, end))
    elif blob_key.startswith(GRAPH_INFO_BLOB_TAG_PREFIX):
        run, graph_id = _parse_graph_info_blob_key(blob_key)
        return json.dumps(self._multiplexer.GraphInfo(run, graph_id))
    elif blob_key.startswith(GRAPH_OP_INFO_BLOB_TAG_PREFIX):
        run, graph_id, op_name = _parse_graph_op_info_blob_key(blob_key)
        return json.dumps(self._multiplexer.GraphOpInfo(run, graph_id, op_name))
    elif blob_key.startswith(SOURCE_FILE_LIST_BLOB_TAG):
        run = _parse_source_file_list_blob_key(blob_key)
        return json.dumps(self._multiplexer.SourceFileList(run))
    elif blob_key.startswith(SOURCE_FILE_BLOB_TAG_PREFIX):
        run, index = _parse_source_file_blob_key(blob_key)
        return json.dumps(self._multiplexer.SourceLines(run, index))
    elif blob_key.startswith(STACK_FRAMES_BLOB_TAG_PREFIX):
        run, stack_frame_ids = _parse_stack_frames_blob_key(blob_key)
        return json.dumps(self._multiplexer.StackFrames(run, stack_frame_ids))
    else:
        raise ValueError('Unrecognized blob_key: %s' % blob_key)