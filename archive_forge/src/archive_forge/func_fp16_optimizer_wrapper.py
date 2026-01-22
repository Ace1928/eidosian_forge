import math
from itertools import chain
from typing import Optional
import parlai.utils.logging as logging
from parlai.utils.misc import error_once
def fp16_optimizer_wrapper(optimizer: torch.optim.Optimizer, verbose: bool=False, dynamic_loss_scale: bool=True, loss_initial_scale: float=2.0 ** 17):
    """
    Wrap the an optimizer with FP16 loss scaling protection.

    Requires apex to be installed. Will throw an ImportError if it is not.

    :param optimizer:
        Any torch optimizer
    :param bool verbose:
        Enables verbose output in the FP16 optimizer. Turning this on can help
        debug when FP16 is underperforming.
    :param bool dynamic_loss_scaling:
        FP16 requires loss scaling to avoid underflows. It is recommended this
        stays on, but advanced users may want it off.
    :param float loss_initial_scale:
        Initial loss scaling. Default chosen empirically, but models with very low
        or high loss values may need this adjusted. Stick with powers of 2.

    :returns:
        An APEX FP16 optimizer. Please note this has different requirements on
        how backward() and step() are called.
    """
    try:
        import apex.fp16_utils
    except ImportError:
        raise ImportError('No fp16 support without apex. Please install it from https://github.com/NVIDIA/apex')
    return apex.fp16_utils.FP16_Optimizer(optimizer, dynamic_loss_scale=dynamic_loss_scale, verbose=verbose, dynamic_loss_args={'init_scale': loss_initial_scale})