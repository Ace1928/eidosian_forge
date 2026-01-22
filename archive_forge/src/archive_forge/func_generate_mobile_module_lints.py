import torch
from enum import Enum
from torch._C import _MobileOptimizerType as MobileOptimizerType
from typing import Optional, Set, List, AnyStr
def generate_mobile_module_lints(script_module: torch.jit.ScriptModule):
    """
    Generate a list of lints for a given torch script module.

    Args:
        script_module: An instance of torch script module with type of ScriptModule.

    Returns:
        lint_map: A list of dictionary that contains modules lints
    """
    if not isinstance(script_module, torch.jit.ScriptModule):
        raise TypeError(f'Got {type(script_module)}, but ScriptModule is expected.')
    lint_list = []
    if not hasattr(script_module, '_generate_bundled_inputs_for_forward'):
        lint_list.append({'name': LintCode.BUNDLED_INPUT.name, 'message': 'No bundled input for forward, please add bundled inputs before saving the module using torch.utils.bundled_inputs.augment_model_with_bundled_inputs.'})
    for name, param in script_module.named_parameters():
        if param.requires_grad:
            lint_list.append({'name': LintCode.REQUIRES_GRAD.name, 'message': f'Param {name} requires grad, please set torch.no_grad() to reduce memory usage and improve computation speed during inference phase.'})
    op_names = torch.jit.export_opnames(script_module)
    for op_name in op_names:
        if 'dropout' in op_name:
            lint_list.append({'name': LintCode.DROPOUT.name, 'message': 'Operator {} exists, remember to call eval() before saving the module.and call torch.utils.mobile_optimizer.optimize_for_mobile to drop dropout operator.'.format(op_name)})
        if 'batch_norm' in op_name:
            lint_list.append({'name': LintCode.BATCHNORM.name, 'message': 'Operator {} exists, remember to call eval() before saving the module and call torch.utils.mobile_optimizer.optimize_for_mobile to drop batch_norm operator.'.format(op_name)})
    return lint_list