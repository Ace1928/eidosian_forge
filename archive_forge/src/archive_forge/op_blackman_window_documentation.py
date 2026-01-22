import numpy as np
from onnx.reference.ops._op_common_window import _CommonWindow
Blankman windowing function.

    Returns :math:`\\omega_n = 0.42 - 0.5 \\cos \\left( \\frac{2\\pi n}{N-1} \\right) + 0.08 \\cos \\left( \\frac{4\\pi n}{N-1} \\right)`
    where *N* is the window length.

    See `blackman_window <https://pytorch.org/docs/stable/generated/torch.blackman_window.html>`_
    