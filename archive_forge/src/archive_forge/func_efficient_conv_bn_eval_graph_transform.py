import torch
import torch.nn as nn
from torch._dynamo.utils import counters
from torch._inductor import config as inductor_config
from torch.func import functional_call
from ..pattern_matcher import CallModuleVarArgs, Match, register_graph_pattern
from .pre_grad import efficient_conv_bn_eval_pass
@register_graph_pattern(CallModuleVarArgs([nn.modules.batchnorm._BatchNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm]), pass_dict=efficient_conv_bn_eval_pass, extra_check=lambda match: not inductor_config.freezing and inductor_config.efficient_conv_bn_eval_fx_passes)
def efficient_conv_bn_eval_graph_transform(match: Match, *args, **kwargs):
    bn_node = match.nodes[0]
    graph = match.graph
    gm = graph.owning_module
    bn_mod = getattr(gm, bn_node.target)
    if not bn_mod.track_running_stats or bn_mod.training:
        return
    if bn_node.args:
        input_node = bn_node.args[0]
    else:
        input_node = bn_node.kwargs['input']
    if input_node.op != 'call_module':
        return
    if not hasattr(gm, input_node.target):
        return
    input_mod = getattr(gm, input_node.target)
    supported_convs = [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]
    if not any((isinstance(input_mod, cls) for cls in supported_convs)):
        return
    conv_node = input_node
    if len(conv_node.users) > 1:
        return
    counters['inductor']['efficient_conv_bn_eval'] += 1
    with graph.inserting_before(conv_node):
        conv_get_node = graph.create_node(op='get_attr', target=conv_node.target, name='get_conv')
        bn_get_node = graph.create_node(op='get_attr', target=bn_node.target, name='get_bn')
        if conv_node.args:
            conv_input = conv_node.args[0]
        else:
            conv_input = conv_node.kwargs['input']
        args = (bn_get_node, conv_get_node, conv_input)
        new_node = graph.create_node(op='call_function', target=efficient_conv_bn_eval, args=args, name='efficient_conv_bn_eval')
    bn_node.replace_all_uses_with(new_node)
    graph.erase_node(bn_node)
    graph.erase_node(conv_node)