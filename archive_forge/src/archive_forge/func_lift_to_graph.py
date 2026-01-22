import collections
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import op_selector
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.util import compat
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export
@tf_export('__internal__.lift_to_graph', v1=[])
def lift_to_graph(tensors, graph, sources=None, disallowed_placeholders=None, add_sources=False, handle_captures=False, base_graph=None, op_map=None):
    """Copies the tensor and all its inputs recursively to the outer graph.

  Args:
    tensors: The Tensors to lift.
    graph: The graph to lift to.
    sources: Optional sequence of nodes to start from. If omitted the whole
      subgraph which feeds into `init_tensor` is lifted.
    disallowed_placeholders: An optional set of ops which may not appear in the
      lifted graph. Defaults to all placeholders.
    add_sources: A boolean indicating whether placeholders which are not in
      sources should be allowed.
    handle_captures: A boolean indicating whether to re-capture s in the new
      graph or simply create a vanilla placeholder.
    base_graph: The graph from which to lift ops. This will be inferred if not
      specified.
    op_map: A map contains all the existing nodes that have been lifted to the
      destination graph, so they won't be lifted and copied again.

  Returns:
    A mapping from ops in the current default graph to ops in `graph`.

  Raises:
    UnliftableError: If a placeholder blocks lifting.
  """
    variable_init_tensors = []
    init_tensors = []
    for tensor in tensors:
        if isinstance(tensor, resource_variable_ops.ResourceVariable):
            variable_init_tensors.append(tensor)
        else:
            init_tensors.append(tensor)
    base_graph = base_graph or init_tensors[0].graph
    op_map = op_map or object_identity.ObjectIdentityDictionary()
    sources = object_identity.ObjectIdentitySet(sources or [])
    visited_ops = set((x.op for x in sources))
    op_outputs = collections.defaultdict(set)
    for init_tensor in init_tensors:
        sources.update(op_selector.map_subgraph(init_tensor=init_tensor, sources=sources, disallowed_placeholders=disallowed_placeholders, visited_ops=visited_ops, op_outputs=op_outputs, add_sources=add_sources))
    ops_to_copy = []
    marked_ops = set([])
    ops_to_visit = [_as_operation(t) for t in init_tensors if not op_outputs[_as_operation(t)]]
    unvisited_ops = set(ops_to_visit)
    while unvisited_ops:
        while ops_to_visit:
            op = ops_to_visit.pop()
            if op in marked_ops:
                continue
            marked_ops.add(op)
            ops_to_copy.append(op)
            for inp in op_selector.graph_inputs(op):
                if inp.type == 'TPUReplicateMetadata':
                    continue
                unvisited_ops.add(inp)
                if all((x in marked_ops for x in op_outputs[inp])) and inp not in sources:
                    ops_to_visit.append(inp)
        unvisited_ops.difference_update(marked_ops)
        if unvisited_ops:
            ops_to_visit.append(next(iter(unvisited_ops)))
    ops_to_copy.sort(key=lambda op: len(op_selector.graph_inputs(op)) == 0)
    captures = []
    inverse_captures = object_identity.ObjectIdentityDictionary()
    internal_captures = []
    if isinstance(base_graph, func_graph.FuncGraph) and isinstance(graph, func_graph.FuncGraph):
        captures = base_graph.captures
        for external_capture, internal_capture in captures:
            inverse_captures[internal_capture] = external_capture
        internal_captures = base_graph.internal_captures
    with graph.as_default():
        for i in variable_init_tensors:
            op_map[i] = i
        source_ops = set()
        for s in internal_captures:
            if s in sources:
                sources.remove(s)
                source_ops.add(s.op)
                _copy_source(s=s, graph=graph, op_map=op_map, handle_captures=handle_captures, inverse_captures=inverse_captures, base_graph=base_graph)
        for s in sources:
            source_ops.add(s.op)
            _copy_source(s=s, graph=graph, op_map=op_map, handle_captures=handle_captures, inverse_captures=inverse_captures, base_graph=base_graph)
        input_mutations = []
        control_mutations = []
        for op in reversed(ops_to_copy):
            if op in source_ops or op in op_map:
                continue
            new_input_mutations, new_control_mutations = _copy_non_source(op=op, graph=graph, op_map=op_map, base_graph=base_graph)
            input_mutations.extend(new_input_mutations)
            control_mutations.extend(new_control_mutations)
        with graph._mutation_lock():
            for mutation in input_mutations:
                mutation.copied_op._update_input(mutation.input_index, op_map[mutation.old_graph_tensor])
            for mutation in control_mutations:
                if mutation.old_graph_op.type == 'TPUReplicateMetadata':
                    continue
                mutation.copied_op._add_control_input(op_map[mutation.old_graph_op])
        return op_map