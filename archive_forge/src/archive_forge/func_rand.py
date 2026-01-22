from ..runtime.jit import jit
from . import core as tl
from . import standard
@jit
def rand(seed, offset, n_rounds: tl.constexpr=N_ROUNDS_DEFAULT):
    """
    Given a :code:`seed` scalar and an :code:`offset` block,
    returns a block of random :code:`float32` in :math:`U(0, 1)`.

    :param seed: The seed for generating random numbers.
    :param offsets: The offsets to generate random numbers for.
    """
    source = randint(seed, offset, n_rounds)
    return uint_to_uniform_float(source)