import numpy as np
from onnx.reference.ops._op_common_window import _CommonWindow
Returns :math:`\\omega_n = \\alpha - \\beta \\cos \\left( \\frac{\\pi n}{N-1} \\right)` where *N* is the window length.

    See `hamming_window <https://pytorch.org/docs/stable/generated/torch.hamming_window.html>`_.
    `alpha=0.54, beta=0.46`
    