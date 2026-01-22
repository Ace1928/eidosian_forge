import copy
import os
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union
import onnx
from onnx import ModelProto
from ..utils import logging
from .transformations_utils import (
def merge_decoders(decoder: Union[ModelProto, Path, str], decoder_with_past: Union[ModelProto, Path, str], graph_name: str='merged', producer_name: str='optimum-onnx', save_path: Optional[Union[str, Path]]=None, strict: bool=True) -> ModelProto:
    """
    Fuses decoder ONNX model and decoder with past ONNX model into one ONNX model with if logic.

    Args:
        decoder (`Union[ModelProto, Path, str]`):
            Decoder ONNX model.
        decoder_with_past (`Union[ModelProto, Path, str]`):
            Decoder with past ONNX model.
        graph_name (`str`, defaults to `"merged"`):
            Name of the parent graph (graph of the control flow node).
        producer_name (`str`, defaults to `"optimum-onnx"`):
            Graph producer name.
        save_path (`Optional[Union[str, Path]]`, defaults to `None`):
            The path to save merged ONNX model. The model will be saved if the path is given.
        strict (`bool`, defaults to `True`):
            When set, the decoder and decoder_with_past are expected to have strictly the same number of outputs. When False,
            the decoder is allowed to have more outputs that decoder_with_past, in which case constant outputs are added to match
            the number of outputs.

    Returns:
        `~onnx.ModelProto`: The fused decoder ONNX model.
    """
    if isinstance(decoder, (str, Path)):
        decoder = Path(decoder).as_posix()
        decoder = onnx.load(decoder)
    if isinstance(decoder_with_past, (str, Path)):
        decoder_with_past = Path(decoder_with_past).as_posix()
        decoder_with_past = onnx.load(decoder_with_past)
    decoder_opset = _get_onnx_opset(decoder)
    decoder_with_past_opset = _get_onnx_opset(decoder_with_past)
    if decoder_opset != decoder_with_past_opset:
        raise ValueError(f"Decoder's opset is {decoder_opset}, but decoder with past's opset is {decoder_with_past_opset}. Make sure having the same opset before merging.")
    _unify_onnx_outputs(decoder, decoder_with_past, strict=strict)
    all_inputs = _get_all_inputs([decoder, decoder_with_past])
    for _, inp in enumerate(all_inputs):
        if inp.name == 'attention_mask':
            if inp.type.tensor_type.shape.dim[1].dim_param != 'sequence_length':
                raise ValueError('Expected attention_mask second axis to be dynamic and named `sequence_length`.')
            inp.type.tensor_type.shape.dim[1].dim_param = 'attention_mask_sequence_length'
    deduplicated_initializers = _deduplicated_cross_model_initializers([decoder, decoder_with_past], suffix=graph_name)
    decoder_initializers = []
    for initializer in decoder.graph.initializer:
        if len(initializer.dims) == 0 or (len(initializer.dims) == 1 and initializer.data_type in [6, 7]):
            decoder_initializers.append(initializer)
    decoder_with_past_initializers = []
    for initializer in decoder_with_past.graph.initializer:
        if len(initializer.dims) == 0 or (len(initializer.dims) == 1 and initializer.data_type in [6, 7]):
            decoder_with_past_initializers.append(initializer)
    no_past_branch = onnx.helper.make_graph(nodes=decoder.graph.node, name='no_past', inputs=[], outputs=decoder.graph.output, initializer=decoder_initializers)
    with_past_branch = onnx.helper.make_graph(nodes=decoder_with_past.graph.node, name='with_past', inputs=[], outputs=decoder_with_past.graph.output, initializer=decoder_with_past_initializers)
    use_cache_branch = onnx.helper.make_tensor_value_info(name='use_cache_branch', elem_type=onnx.TensorProto.BOOL, shape=[1])
    if_node = onnx.helper.make_node('If', inputs=['use_cache_branch'], outputs=[output.name for output in no_past_branch.output], name='optimum::if', then_branch=with_past_branch, else_branch=no_past_branch)
    merged_graph = onnx.helper.make_graph(nodes=[if_node], name=graph_name, inputs=all_inputs + [use_cache_branch], outputs=no_past_branch.output, initializer=deduplicated_initializers)
    opset_imports = []
    opset_domains = set()
    for opset_import in list(decoder.opset_import) + list(decoder_with_past.opset_import):
        if opset_import.domain not in opset_domains:
            opset_imports.append(opset_import)
            opset_domains.add(opset_import.domain)
    merged_model = onnx.helper.make_model(merged_graph, producer_name=producer_name, opset_imports=opset_imports)
    check_and_save_model(merged_model, save_path=save_path)
    return merged_model