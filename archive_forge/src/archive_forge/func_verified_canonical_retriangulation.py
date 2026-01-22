from ..sage_helper import _within_sage, sage_method
from .cuspCrossSection import RealCuspCrossSection
from .squareExtensions import find_shapes_as_complex_sqrt_lin_combinations
from . import verifyHyperbolicity
from . import exceptions
from ..exceptions import SnapPeaFatalError
from ..snap import t3mlite as t3m
@sage_method
def verified_canonical_retriangulation(M, interval_bits_precs=default_interval_bits_precs, exact_bits_prec_and_degrees=default_exact_bits_prec_and_degrees, verbose=False):
    """
    Given some triangulation of a cusped (possibly non-orientable) manifold ``M``,
    return its canonical retriangulation. Return ``None`` if it could not certify
    the result.

    To compute the canonical retriangulation, it first prepares the manifold
    (filling all Dehn-filled cusps and trying to find a proto-canonical
    triangulation).
    It then tries to certify the canonical triangulation using interval
    arithmetics. If this fails, it uses snap (using `LLL-algorithm
    <http://en.wikipedia.org/wiki/Lenstra%E2%80%93Lenstra%E2%80%93Lov%C3%A1sz_lattice_basis_reduction_algorithm>`_)
    to guess
    exact representations of the shapes in the shape field and then certifies
    that it found the proto-canonical triangulation and determines the
    transparent faces to construct the canonical retriangulation.

    The optional arguments are:

    - ``interval_bits_precs``:
      a list of precisions used to try to
      certify the canonical triangulation using intervals. By default, it
      first tries to certify using 53 bits precision. If it failed, it tries
      212 bits precision next. If it failed again, it moves on to trying exact
      arithmetics.

    - ``exact_bits_prec_and_degrees``:
      a list of pairs (precision, maximal degree) used when the LLL-algorithm
      is trying to find the defining polynomial of the shape field.
      Similar to ``interval_bits_precs``, each pair is tried until we succeed.

    - ``verbose``:
      If ``True``, print out additional information.

    The exact arithmetics can take a long time. To circumvent it, use
    ``exact_bits_prec_and_degrees = None``.

    More information on the canonical retriangulation can be found in the
    SnapPea kernel ``canonize_part_2.c`` and in Section 3.1 of
    `Fominykh, Garoufalidis, Goerner, Tarkaev, Vesnin <http://arxiv.org/abs/1502.00383>`_.

    Canonical cell decomposition of ``m004`` has 2 tetrahedral cells::

       sage: from snappy import Manifold
       sage: M = Manifold("m004")
       sage: K = verified_canonical_retriangulation(M)
       sage: K.has_finite_vertices()
       False
       sage: K.num_tetrahedra()
       2

    Canonical cell decomposition of ``m137`` is not tetrahedral::

       sage: M = Manifold("m137")
       sage: K = verified_canonical_retriangulation(M)
       sage: K.has_finite_vertices()
       True
       sage: K.num_tetrahedra()
       18

    Canonical cell decomposition of ``m412`` is a cube and has exactly 8
    symmetries::

       sage: M = Manifold("m412")
       sage: K = verified_canonical_retriangulation(M)
       sage: K.has_finite_vertices()
       True
       sage: K.num_tetrahedra()
       12
       sage: len(K.isomorphisms_to(K))
       8

    `Burton's example <http://arxiv.org/abs/1311.7615>`_ of ``x101`` and ``x103`` which are actually isometric but
    SnapPea fails to show so. We certify the canonical retriangulation and
    find them isomorphic::

       sage: M = Manifold('x101'); K = verified_canonical_retriangulation(M)
       sage: N = Manifold('x103'); L = verified_canonical_retriangulation(N)
       sage: len(K.isomorphisms_to(L)) > 0
       True

    Avoid potentially expensive exact arithmetics (return ``None`` because it has
    non-tetrahedral cells so interval arithmetics can't certify it)::

       sage: M = Manifold("m412")
       sage: verified_canonical_retriangulation(M, exact_bits_prec_and_degrees = None)
    """
    tries_penalty_left = _max_tries_verify_penalty
    while tries_penalty_left > 0:
        try:
            return _verified_canonical_retriangulation(M, interval_bits_precs, exact_bits_prec_and_degrees, verbose)
        except (ZeroDivisionError, exceptions.TiltProvenPositiveNumericalVerifyError, exceptions.EdgeEquationExactVerifyError) as e:
            if verbose:
                _print_exception(e)
                print("Failure: In verification of result of SnapPea kernel's proto_canonize", end='')
                if isinstance(e, ZeroDivisionError):
                    print(' probably due to flat tetrahedra.')
                if isinstance(e, exceptions.TiltProvenPositiveNumericalVerifyError):
                    print(' due to provably positive tilts.')
                if isinstance(e, exceptions.EdgeEquationExactVerifyError):
                    print(' probably due to snap giving wrong number field.')
                print('Next step: Retrying with randomized triangulation.')
            M = M.copy()
            M.randomize()
            if isinstance(e, ZeroDivisionError):
                tries_penalty_left -= 1
            else:
                tries_penalty_left -= 3
        except exceptions.VerifyErrorBase as e:
            if verbose:
                _print_exception(e)
                print("Failure: In verification of result of SnapPea kernel's proto_canonize.")
                print('Next step: Give up.')
            return None
    return None