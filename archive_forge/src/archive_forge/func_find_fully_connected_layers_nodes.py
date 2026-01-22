from typing import List
from onnxruntime.transformers.onnx_model import OnnxModel
def find_fully_connected_layers_nodes(model: OnnxModel) -> List[List[str]]:
    adds = model.get_nodes_by_op_type('Add')
    fc = list(filter(lambda graph: graph[1] is not None, ((add, model.match_parent(add, 'MatMul')) for add in adds)))
    return fc