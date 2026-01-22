import itertools
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch._logging import getArtifactLogger
from torch._subclasses.functional_tensor import FunctionalTensor
from torch.fx.experimental.symbolic_shapes import is_concrete_int
from .functional_utils import _get_mutation_type
from .schemas import (
from .utils import strict_zip
def create_graph_signature(fx_g: torch.fx.GraphModule, fw_metadata: ViewAndMutationMeta, in_spec: pytree.TreeSpec, out_spec: pytree.TreeSpec, *, user_args_flat: List[Tensor], params_and_buffers_flat: List[Tensor], param_names: List[str], buffer_names: List[str], trace_joint: bool, num_user_fw_outs: Optional[int], loss_index: Optional[int]) -> GraphSignature:
    graph_input_names = _graph_input_names(fx_g)
    graph_output_names = _graph_output_names(fx_g)
    num_params_buffers = len(param_names) + len(buffer_names)
    num_user_args = len(graph_input_names) - num_params_buffers
    if trace_joint:
        assert num_user_fw_outs is not None
        num_fw_outs = num_user_fw_outs + fw_metadata.num_mutated_inp_runtime_indices
        backward_output_names = graph_output_names[num_fw_outs:]
        grad_index = itertools.count(0)
        gradients_to_parameters = {backward_output_names[next(grad_index)]: param_names[i] for i, param in enumerate(params_and_buffers_flat) if param.requires_grad}
        gradients_to_user_inputs = {backward_output_names[next(grad_index)]: graph_input_names[i + len(params_and_buffers_flat)] for i, user_input in enumerate(user_args_flat) if user_input.requires_grad}
        assert len(gradients_to_parameters) + len(gradients_to_user_inputs) == len(backward_output_names)
        backward_signature = BackwardSignature(gradients_to_parameters, gradients_to_user_inputs, graph_output_names[loss_index])
    else:
        backward_signature = None
        num_user_fw_outs = len(graph_output_names) - fw_metadata.num_mutated_inp_runtime_indices
    return GraphSignature.from_tracing_metadata(in_spec=in_spec, out_spec=out_spec, graph_input_names=graph_input_names, graph_output_names=graph_output_names, view_mutation_metadata=fw_metadata, named_parameters=param_names, named_buffers=buffer_names, num_user_inputs=num_user_args, num_user_outputs=num_user_fw_outs, loss_index=loss_index, backward_signature=backward_signature)