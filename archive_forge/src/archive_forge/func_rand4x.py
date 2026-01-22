from ..runtime.jit import jit
from . import core as tl
from . import standard
@jit
def rand4x(seed, offsets, n_rounds: tl.constexpr=N_ROUNDS_DEFAULT):
    """
    Given a :code:`seed` scalar and an :code:`offsets` block,
    returns 4 blocks of random :code:`float32` in :math:`U(0, 1)`.

    :param seed: The seed for generating random numbers.
    :param offsets: The offsets to generate random numbers for.
    """
    i1, i2, i3, i4 = randint4x(seed, offsets, n_rounds)
    u1 = uint_to_uniform_float(i1)
    u2 = uint_to_uniform_float(i2)
    u3 = uint_to_uniform_float(i3)
    u4 = uint_to_uniform_float(i4)
    return (u1, u2, u3, u4)