import numbers
from collections import defaultdict
from fractions import Fraction
import numpy as np
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.variable import Variable
def fracify(a, max_denom: int=1024, force_dyad: bool=False):
    """ Return a valid fractional weight tuple (and its dyadic completion)
        to represent the weights given by ``a``.

        When the input tuple contains only integers and fractions,
        ``fracify`` will try to represent the weights exactly.

    Parameters
    ----------
    a : Sequence
        Sequence of numbers (ints, floats, or Fractions) to be represented
        with fractional weights.

    max_denom : int
        The maximum denominator allowed for the fractional representation.
        When the fractional representation is not exact, increasing
        ``max_denom`` will typically give a better approximation.

        Note that ``max_denom`` is actually replaced with the largest power
        of 2 >= ``max_denom``.

    force_dyad : bool
        If ``True``, we force w to be a dyadic representation so that ``w == w_dyad``.
        This means that ``w_dyad`` does not need an extra dummy variable.
        In some cases, this may reduce the number of second-order cones needed to
        represent ``w``.

    Returns
    -------
    w : tuple
        Approximation of ``a/sum(a)`` as a tuple of fractions.

    w_dyad : tuple
        The dyadic completion of ``w``.

        That is, if w has fractions with denominators that are not a power of 2,
        and ``len(w) == n`` then w_dyad has length n+1, dyadic fractions for elements,
        and ``w_dyad[:-1]/w_dyad[n] == w``.
        # ^ That isn't always possible, is it?

        Alternatively, the ratios between the
        first n elements of ``w_dyad`` are equal to the corresponding ratios between
        the n elements of ``w``.

        The dyadic completion of w is needed to represent the weighted geometric
        mean with weights ``w`` as a collection of second-order cones.

        The appended element of ``w_dyad`` is typically a dummy variable.

    Examples
    --------
    >>> w, w_dyad = fracify([1, 2, 3])
    >>> w
    (Fraction(1, 6), Fraction(1, 3), Fraction(1, 2))
    >>> w_dyad
    (Fraction(1, 8), Fraction(1, 4), Fraction(3, 8), Fraction(1, 4))

    >>> w, w_dyad = fracify((1, 1, 1, 1, 1))
    >>> w
    (Fraction(1, 5), Fraction(1, 5), Fraction(1, 5), Fraction(1, 5), Fraction(1, 5))
    >>> w_dyad
    (Fraction(1, 8), Fraction(1, 8), Fraction(1, 8), Fraction(1, 8), Fraction(1, 8), Fraction(3, 8))

    >>> w, w_dyad = fracify([.23, .56, .87])
    >>> w
    (Fraction(23, 166), Fraction(28, 83), Fraction(87, 166))
    >>> w_dyad
    (Fraction(23, 256), Fraction(7, 32), Fraction(87, 256), Fraction(45, 128))

    >>> w, w_dyad = fracify([3, Fraction(1, 2), Fraction(3, 5)])
    >>> w
    (Fraction(30, 41), Fraction(5, 41), Fraction(6, 41))
    >>> w_dyad
    (Fraction(15, 32), Fraction(5, 64), Fraction(3, 32), Fraction(23, 64))

    Can also mix integer, Fraction, and floating point types.

    >>> w, w_dyad = fracify([3.4, 8, Fraction(3, 2)])
    >>> w
    (Fraction(34, 129), Fraction(80, 129), Fraction(5, 43))
    >>> w_dyad
    (Fraction(17, 128), Fraction(5, 16), Fraction(15, 256), Fraction(127, 256))

    Forcing w to be dyadic makes it its own dyadic completion.

    >>> w, w_dyad = fracify([3.4, 8, Fraction(3, 2)], force_dyad=True)
    >>> w
    (Fraction(135, 512), Fraction(635, 1024), Fraction(119, 1024))
    >>> w_dyad
    (Fraction(135, 512), Fraction(635, 1024), Fraction(119, 1024))

    A standard basis unit vector should yield itself.

    >>> w, w_dyad = fracify((0, 0.0, 1.0))
    >>> w
    (Fraction(0, 1), Fraction(0, 1), Fraction(1, 1))
    >>> w_dyad
    (Fraction(0, 1), Fraction(0, 1), Fraction(1, 1))

    A dyadic weight vector should also yield itself.

    >>> a = (Fraction(1,2), Fraction(1,8), Fraction(3,8))
    >>> w, w_dyad = fracify(a)
    >>> a == w == w_dyad
    True

    Be careful when converting floating points to fractions.

    >>> a = (Fraction(.9), Fraction(.1))
    >>> w, w_dyad = fracify(a)
    Traceback (most recent call last):
    ...
    ValueError: Can't reliably represent the input weight vector.
    Try increasing `max_denom` or checking the denominators of your input fractions.

    The error here is because ``Fraction(.9)`` and ``Fraction(.1)``
    evaluate to ``(Fraction(8106479329266893, 9007199254740992)`` and
    ``Fraction(3602879701896397, 36028797018963968))``.

    """
    if any((v < 0 for v in a)):
        raise ValueError('Input powers must be nonnegative.')
    if not (isinstance(max_denom, numbers.Integral) and max_denom > 0):
        raise ValueError('Input denominator must be an integer.')
    if isinstance(a, np.ndarray):
        a = a.tolist()
    max_denom = next_pow2(max_denom)
    total = sum(a)
    if force_dyad is True:
        w_frac = make_frac(a, max_denom)
    elif all((isinstance(v, (numbers.Integral, Fraction)) for v in a)):
        w_frac = tuple((Fraction(v, total) for v in a))
        d = max((v.denominator for v in w_frac))
        if d > max_denom:
            raise ValueError(__EXCEED_DENOMINATOR_LIMIT__)
    else:
        w_frac = tuple((Fraction(float(v) / total).limit_denominator(max_denom) for v in a))
        if sum(w_frac) != 1:
            w_frac = make_frac(a, max_denom)
    w_dyad = dyad_completion(w_frac)
    if max((v.denominator for v in w_dyad)) > max_denom:
        raise ValueError(__EXCEED_DENOMINATOR_LIMIT__)
    return (w_frac, w_dyad)