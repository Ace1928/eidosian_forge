import numpy as np
from .... import symbol
from .... import ndarray as nd
from ....base import string_types
from ._import_helper import _convert_map as convert_map
def from_onnx(self, graph, opset_version):
    """Construct symbol from onnx graph.

        Parameters
        ----------
        graph : onnx protobuf object
            The loaded onnx graph

        Returns
        -------
        sym :symbol.Symbol
            The returned mxnet symbol
        params : dict
            A dict of name: nd.array pairs, used as pretrained weights
        """
    self.opset_version = opset_version
    self.model_metadata = self.get_graph_metadata(graph)
    for init_tensor in graph.initializer:
        if not init_tensor.name.strip():
            raise ValueError("Tensor's name is required.")
        self._params[init_tensor.name] = self._parse_array(init_tensor)
    for i in graph.input:
        if i.name in self._params:
            self._nodes[i.name] = symbol.Variable(name=i.name, shape=self._params[i.name].shape)
        else:
            self._nodes[i.name] = symbol.Variable(name=i.name)
    for node in graph.node:
        op_name = node.op_type
        node_name = node.name.strip()
        node_name = node_name if node_name else None
        onnx_attr = self._parse_attr(node.attribute)
        inputs = [self._nodes[i] for i in node.input]
        mxnet_sym = self._convert_operator(node_name, op_name, onnx_attr, inputs)
        for k, i in zip(list(node.output), range(len(mxnet_sym.list_outputs()))):
            self._nodes[k] = mxnet_sym[i]
        for args in mxnet_sym.list_arguments():
            if args in self._params:
                self.arg_dict.update({args: nd.array(self._params[args])})
        for aux in mxnet_sym.list_auxiliary_states():
            if aux in self._params:
                self.aux_dict.update({aux: nd.array(self._params[aux])})
    out = [self._nodes[i.name] for i in graph.output]
    if len(out) > 1:
        out = symbol.Group(out)
    else:
        out = out[0]
    return (out, self.arg_dict, self.aux_dict)