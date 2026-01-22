from contextlib import contextmanager
import packaging.version
import torch
import transformers
def dequantize_bnb_weight(weight: torch.nn.Parameter, state=None):
    """
    Helper function to dequantize 4bit or 8bit bnb weights.

    If the weight is not a bnb quantized weight, it will be returned as is.
    """
    if not isinstance(weight, torch.nn.Parameter):
        raise TypeError(f'Input weight should be of type nn.Parameter, got {type(weight)} instead')
    cls_name = weight.__class__.__name__
    if cls_name not in ('Params4bit', 'Int8Params'):
        return weight
    import bitsandbytes as bnb
    if cls_name == 'Params4bit':
        return bnb.functional.dequantize_4bit(weight.data, weight.quant_state)
    if state.SCB is None:
        state.SCB = weight.SCB
    im = torch.eye(weight.data.shape[-1]).contiguous().half().to(weight.device)
    im, imt, SCim, SCimt, coo_tensorim = bnb.functional.double_quant(im)
    im, Sim = bnb.functional.transform(im, 'col32')
    if state.CxB is None:
        state.CxB, state.SB = bnb.functional.transform(weight.data, to_order=state.formatB)
    out32, Sout32 = bnb.functional.igemmlt(im, state.CxB, Sim, state.SB)
    return bnb.functional.mm_dequant(out32, Sout32, SCim, state.SCB, bias=None).t()