from ..runtime.jit import jit
from . import core as tl
from . import standard

    Given a :code:`seed` scalar and an :code:`offset` block,
    returns 4 blocks of random :code:`float32` in :math:`\mathcal{N}(0, 1)`.

    :param seed: The seed for generating random numbers.
    :param offsets: The offsets to generate random numbers for.
    