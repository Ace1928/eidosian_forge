import re
def watch_graph(run_options, graph, debug_ops='DebugIdentity', debug_urls=None, node_name_regex_allowlist=None, op_type_regex_allowlist=None, tensor_dtype_regex_allowlist=None, tolerate_debug_op_creation_failures=False, global_step=-1, reset_disk_byte_usage=False):
    """Add debug watches to `RunOptions` for a TensorFlow graph.

  To watch all `Tensor`s on the graph, let both `node_name_regex_allowlist`
  and `op_type_regex_allowlist` be the default (`None`).

  N.B.:
    1. Under certain circumstances, the `Tensor` may not get actually watched
      (e.g., if the node of the `Tensor` is constant-folded during runtime).
    2. For debugging purposes, the `parallel_iteration` attribute of all
      `tf.while_loop`s in the graph are set to 1 to prevent any node from
      being executed multiple times concurrently. This change does not affect
      subsequent non-debugged runs of the same `tf.while_loop`s.


  Args:
    run_options: An instance of `config_pb2.RunOptions` to be modified.
    graph: An instance of `ops.Graph`.
    debug_ops: (`str` or `list` of `str`) name(s) of the debug op(s) to use.
    debug_urls: URLs to send debug values to. Can be a list of strings,
      a single string, or None. The case of a single string is equivalent to
      a list consisting of a single string, e.g., `file:///tmp/tfdbg_dump_1`,
      `grpc://localhost:12345`.
      For debug op types with customizable attributes, each debug op name string
      can optionally contain a list of attribute names, in the syntax of:
        debug_op_name(attr_name_1=attr_value_1;attr_name_2=attr_value_2;...)
    node_name_regex_allowlist: Regular-expression allowlist for node_name,
      e.g., `"(weight_[0-9]+|bias_.*)"`
    op_type_regex_allowlist: Regular-expression allowlist for the op type of
      nodes, e.g., `"(Variable|Add)"`.
      If both `node_name_regex_allowlist` and `op_type_regex_allowlist`
      are set, the two filtering operations will occur in a logical `AND`
      relation. In other words, a node will be included if and only if it
      hits both allowlists.
    tensor_dtype_regex_allowlist: Regular-expression allowlist for Tensor
      data type, e.g., `"^int.*"`.
      This allowlist operates in logical `AND` relations to the two allowlists
      above.
    tolerate_debug_op_creation_failures: (`bool`) whether debug op creation
      failures (e.g., due to dtype incompatibility) are to be tolerated by not
      throwing exceptions.
    global_step: (`int`) Optional global_step count for this debug tensor
      watch.
    reset_disk_byte_usage: (`bool`) whether to reset the tracked disk byte
      usage to zero (default: `False`).
  """
    if not debug_ops:
        raise ValueError('debug_ops must not be empty or None.')
    if not debug_urls:
        raise ValueError('debug_urls must not be empty or None.')
    if isinstance(debug_ops, str):
        debug_ops = [debug_ops]
    node_name_pattern = re.compile(node_name_regex_allowlist) if node_name_regex_allowlist else None
    op_type_pattern = re.compile(op_type_regex_allowlist) if op_type_regex_allowlist else None
    tensor_dtype_pattern = re.compile(tensor_dtype_regex_allowlist) if tensor_dtype_regex_allowlist else None
    ops = graph.get_operations()
    for op in ops:
        if not op.outputs:
            continue
        node_name = op.name
        op_type = op.type
        if node_name_pattern and (not node_name_pattern.match(node_name)):
            continue
        if op_type_pattern and (not op_type_pattern.match(op_type)):
            continue
        for slot in range(len(op.outputs)):
            if tensor_dtype_pattern and (not tensor_dtype_pattern.match(op.outputs[slot].dtype.name)):
                continue
            add_debug_tensor_watch(run_options, node_name, output_slot=slot, debug_ops=debug_ops, debug_urls=debug_urls, tolerate_debug_op_creation_failures=tolerate_debug_op_creation_failures, global_step=global_step)
    use_node_name_wildcard = not node_name_pattern and (not op_type_pattern) and (not tensor_dtype_pattern)
    if use_node_name_wildcard:
        add_debug_tensor_watch(run_options, '*', output_slot=-1, debug_ops=debug_ops, debug_urls=debug_urls, tolerate_debug_op_creation_failures=tolerate_debug_op_creation_failures, global_step=global_step)
    run_options.debug_options.reset_disk_byte_usage = reset_disk_byte_usage