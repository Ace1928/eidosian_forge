from typing import List, Optional, Union
import numpy as np
from onnx import (
from onnx.helper import (
from onnx.numpy_helper import from_array
def replace_initializer_by_constant_of_shape(onx: Union[FunctionProto, GraphProto, ModelProto], threshold: int=128, ir_version: Optional[int]=None, use_range: bool=False, value_constant_of_shape: float=0.5):
    """Replace initializers or constant node by nodes *ConstantOfShape* to reduce the size.

    This reduce the cost to write a unit test about a specific graph structure.

    Args:
        onx: ModelProto
        threshold: every initializer under this threshold is not
            impacted
        ir_version: initializer must be specified as input for
            `ir_version <= 3`, this must be specified if onx is
            :class:`FunctionProto` or :class:`GraphProto`
        use_range: if uses operator *Range* instead of *ConstantOfShape*
            to avoid constant tensors
        value_constant_of_shape: value to use as a value for all nodes
            *ConstantOfShape*, a high value may produce nan or inf
            predictions

    Returns:
        onx, modified ModelProto

    The function is designed so that the function can be reapplied on a modified model
    and either replace *ConstantOfShape* with *Range* operators, either replace the fill value
    for every *ConstantOfShape*.
    """
    if isinstance(onx, FunctionProto):
        modified = False
        new_nodes: List[NodeProto] = []
        for node in onx.node:
            if node.op_type == 'Constant':
                cst_nodes = _replace_constant(node, threshold, value_constant_of_shape)
                if len(cst_nodes) == 2:
                    modified = True
                new_nodes.extend(cst_nodes)
                continue
            new_nodes.append(node)
        if modified:
            new_onx = make_function(onx.domain, onx.name, onx.input, onx.output, new_nodes, opset_imports=onx.opset_import)
            if use_range:
                return _replace_constant_of_shape_with_range(new_onx)
            if value_constant_of_shape != 1:
                return _replace_constant_of_shape_value(new_onx, value_constant_of_shape)
            return new_onx
        if use_range:
            return _replace_constant_of_shape_with_range(onx)
        if value_constant_of_shape != 1:
            return _replace_constant_of_shape_value(onx, value_constant_of_shape)
        return onx
    if isinstance(onx, ModelProto):
        new_graph = replace_initializer_by_constant_of_shape(onx.graph, ir_version=ir_version or onx.ir_version, threshold=threshold, use_range=use_range, value_constant_of_shape=value_constant_of_shape)
        new_functions = [replace_initializer_by_constant_of_shape(f, threshold=threshold, ir_version=ir_version or onx.ir_version, use_range=use_range, value_constant_of_shape=value_constant_of_shape) for f in onx.functions]
        model = make_model(new_graph, functions=new_functions, producer_name=onx.producer_name, producer_version=onx.producer_version, ir_version=ir_version or onx.ir_version, doc_string=onx.doc_string, domain=onx.domain, model_version=onx.model_version)
        if len(onx.metadata_props) > 0:
            values = {p.key: p.value for p in onx.metadata_props}
            set_model_props(model, values)
        del model.opset_import[:]
        for oimp in onx.opset_import:
            op_set = model.opset_import.add()
            if oimp.domain == '' and oimp.version < 11 and use_range:
                raise RuntimeError(f'Range was introduced in opset 11 but opset is {oimp.version}.')
            if oimp.domain == '' and oimp.version < 9:
                raise RuntimeError(f'ConstantOfShape was introduced in opset 9 but opset is {oimp.version}.')
            op_set.domain = oimp.domain
            op_set.version = oimp.version
        return model
    if not isinstance(onx, GraphProto):
        raise TypeError(f'onx should be a GraphProto at this stage not {type(onx)}.')
    n_modifications = 0
    new_nodes = []
    removed = set()
    additional_inputs = []
    new_inits: List[TensorProto] = []
    for init in onx.initializer:
        dims = tuple(init.dims)
        size = np.prod(dims)
        if size <= threshold:
            new_inits.append(init)
            continue
        n_modifications += 1
        new_name = f'{init.name}__SHAPE'
        new_inits.append(from_array(np.array(list(dims), dtype=np.int64), name=new_name))
        dtype = tensor_dtype_to_np_dtype(init.data_type)
        node = make_node('ConstantOfShape', [new_name], [init.name], value=from_array(np.array([0.5], dtype=dtype)))
        new_nodes.append(node)
        removed.add(init.name)
        if ir_version is not None and ir_version <= 3:
            additional_inputs.append(make_tensor_value_info(new_name, TensorProto.INT64, [len(dims)]))
    new_sparse_inits: List[SparseTensorProto] = []
    for sp_init in onx.sparse_initializer:
        dims = tuple(sp_init.dims)
        size = np.prod(dims)
        if size <= threshold:
            new_sparse_inits.append(sp_init)
            continue
        raise NotImplementedError(f'This feature is not yet implemented for a sparse initializer (indices.name={sp_init.indices.name!r}, values.name={sp_init.values.name!r}).')
    for node in onx.node:
        if node.op_type == 'Constant':
            shape_nodes = _replace_constant(node, threshold, value_constant_of_shape)
            if len(shape_nodes) == 2:
                n_modifications += 1
            new_nodes.extend(shape_nodes)
            continue
        modified = False
        atts = []
        for att in node.attribute:
            if att.type == AttributeProto.GRAPH and hasattr(att, 'g') and (att.g is not None):
                g = replace_initializer_by_constant_of_shape(att.g, threshold=threshold, ir_version=ir_version, use_range=use_range, value_constant_of_shape=value_constant_of_shape)
                if id(g) != id(att.g):
                    modified = True
                    att = make_attribute(att.name, g)
            atts.append(att)
        if modified:
            new_node = make_node(node.op_type, node.input, node.output)
            new_node.attribute.extend(atts)
            new_nodes.append(new_node)
            n_modifications += 1
        else:
            new_nodes.append(node)
    if n_modifications > 0:
        graph = make_graph(new_nodes, onx.name, [i for i in onx.input if i.name not in removed] + additional_inputs, onx.output, initializer=new_inits, sparse_initializer=new_sparse_inits)
        if use_range:
            return _replace_constant_of_shape_with_range(graph)
        if value_constant_of_shape != 1:
            return _replace_constant_of_shape_value(graph, value_constant_of_shape)
        return graph
    if use_range:
        return _replace_constant_of_shape_with_range(onx)
    if value_constant_of_shape != 1:
        return _replace_constant_of_shape_value(onx, value_constant_of_shape)
    return onx