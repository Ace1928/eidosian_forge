from . import matrix
from . import homology
from .ptolemyObstructionClass import PtolemyObstructionClass
from .ptolemyGeneralizedObstructionClass import PtolemyGeneralizedObstructionClass
from .ptolemyVariety import PtolemyVariety
from .utilities import MethodMappingList
def get_ptolemy_variety(manifold, N, obstruction_class=None, simplify=True, eliminate_fixed_ptolemys=False):
    """
    Generates Ptolemy variety as described in
    (1) Garoufalidis, Thurston, Zickert
    The Complex Volume of SL(n,C)-Representations of 3-Manifolds
    http://arxiv.org/abs/1111.2828

    (2) Garoufalidis, Goerner, Zickert:
    Gluing Equations for PGL(n,C)-Representations of 3-Manifolds
    http://arxiv.org/abs/1207.6711

    The variety can be exported to magma or sage and solved there. The
    solutions can be processed to compute invariants. See below.

    === Arguments ===

    N --- which SL(N,C) we want the variety.

    obstruction_class --- class from Definition 1.7 of (1).
    None for trivial class or a value returned from get_ptolemy_obstruction_classes.
    Short cuts: obstruction_class = 'all' returns a list of Ptolemy varieties
    for each obstruction. For easier iteration, can set obstruction_class to
    an integer.

    simplify --- boolean to indicate whether to simplify the equations which
    significantly reduces the number of variables.
    Simplifying means that several identified Ptolemy coordinates x = y = z = ...
    are eliminated instead of adding relations x - y = 0, y - z = 0, ...
    Defaults to True.

    eliminate_fixed_ptolemys --- boolean to indicate whether to eliminate
    the Ptolemy coordinates that are set to 1 for fixing the decoration.
    Even though this simplifies the resulting representation, setting it to
    True can cause magma to run longer when finding a Groebner basis.
    Defaults to False.

    === Examples for 4_1 ===

    >>> from snappy import Manifold
    >>> M = Manifold("4_1")

    Get the varieties for all obstruction classes at once (use
    help(varieties[0]) for more information):

    >>> varieties = get_ptolemy_variety(M, N = 2, obstruction_class = "all")

    Print the variety as an ideal (sage object) for the non-trivial class:

    >>> varieties[1].ideal    #doctest: +SKIP
    Ideal (-c_0011_0^2 + c_0011_0*c_0101_0 + c_0101_0^2, -c_0011_0^2 - c_0011_0*c_0101_0 + c_0101_0^2, c_0011_0 - 1) of Multivariate Polynomial Ring in c_0011_0, c_0101_0 over Rational Field
    (skip doctest because example only works in sage and not plain python)

    >>> for eqn in varieties[1].equations:
    ...     print("    ", eqn)
         - c_0011_0 * c_0101_0 + c_0011_0^2 + c_0101_0^2
         c_0011_0 * c_0101_0 - c_0011_0^2 - c_0101_0^2
         - 1 + c_0011_0

    Generate a magma input to compute Groebner basis for N = 3:

    >>> p = get_ptolemy_variety(M, N = 3)
    >>> s = p.to_magma()

    The beginning of the magma input

    >>> print(s.strip())       #doctest: +ELLIPSIS
    // Setting up the Polynomial ring and ideal
    <BLANKLINE>
    R<c_0012_0, c_0012_1, c_0102_0, c_0111_0, c_0201_0, c_1011_0, c_1011_1, c_1101_0> := PolynomialRing(RationalField(), 8, "grevlex");
    MyIdeal := ideal<R |
              c_0012_0 * c_1101_0 + c_0102_0 * c_0111_0 - c_0102_0 * c_1011_0,
    ...



    === If you have a magma installation ===

    Call p.compute_solutions() to automatically call magma on the above output
    and produce exact solutions!!!

    >>> try:
    ...     sols = p.compute_solutions(verbose)
    ... except:
    ...     sols = None     # magma failed, use precomputed output instead

    === If you do not have a magma installation ===

    Load a precomputed example from magma which is provided with the package:

    >>> from snappy.ptolemy.processMagmaFile import _magma_output_for_4_1__sl3, solutions_from_magma, triangulation_from_magma
    >>> print(_magma_output_for_4_1__sl3)      #doctest: +ELLIPSIS
    <BLANKLINE>
    ==TRIANGULATION=BEGINS==
    % Triangulation
    4_1
    geometric_solution  2.02988321
    oriented_manifold
    ...

    Recover the original trigangulation:
    >>> M = triangulation_from_magma(_magma_output_for_4_1__sl3)
    >>> M.is_isometric_to(Manifold("4_1"))
    True

    Parse the file and produce solutions:

    >>> if sols is None:    # calling magma failed, so use precomputed example
    ...     sols = solutions_from_magma(_magma_output_for_4_1__sl3)

    === Continue here whether you have or do not have magma ===

    Pick the first solution of the three different solutions (up to Galois
    conjugates):

    >>> len(sols)
    3
    >>> solution = sols[0]

    Read the exact value for c_1020_0 (help(solution) for more information
    on how to compute cross ratios, volumes and other invariants):

    >>> solution['c_1020_0']
    Mod(-1/2*x - 3/2, x^2 + 3*x + 4)

    Example of simplified vs non-simplified variety:

    >>> simplified = get_ptolemy_variety(M, N = 4, obstruction_class = 1)
    >>> full = get_ptolemy_variety(M, N = 4, obstruction_class = 1, simplify = False)
    >>> len(simplified.variables), len(full.variables)
    (21, 63)
    >>> len(simplified.equations), len(full.equations)
    (24, 72)

    === ONLY DOCTESTS, NOT PART OF DOCUMENTATION ===

    >>> varieties = get_ptolemy_variety(M, N = 2, obstruction_class = "all", eliminate_fixed_ptolemys = True)

    >>> for eqn in varieties[1].equations:
    ...     print("    ", eqn)
         1 - c_0101_0 + c_0101_0^2
         - 1 + c_0101_0 - c_0101_0^2

    >>> p = get_ptolemy_variety(M, N = 3, eliminate_fixed_ptolemys = True)
    >>> s = p.to_magma()
    >>> print(s.split('ring and ideal')[1].strip())          #doctest: +ELLIPSIS
    R<c_0012_1, c_0102_0, c_0201_0, c_1011_0, c_1011_1, c_1101_0> := PolynomialRing(RationalField(), 6, "grevlex");
    MyIdeal := ideal<R |
              c_0102_0 - c_0102_0 * c_1011_0 + c_1101_0,
        ...

    """
    if hasattr(manifold, 'cusp_info'):
        if False in manifold.cusp_info('is_complete'):
            raise Exception('Dehn fillings not supported by Ptolemy variety')
    N = int(N)
    if obstruction_class is None or isinstance(obstruction_class, PtolemyObstructionClass) or isinstance(obstruction_class, PtolemyGeneralizedObstructionClass):
        return PtolemyVariety(manifold, N, obstruction_class, simplify=simplify, eliminate_fixed_ptolemys=eliminate_fixed_ptolemys)
    list_obstruction_classes = False
    if obstruction_class == 'all_original':
        if N % 2 == 0:
            obstruction_classes = get_ptolemy_obstruction_classes(manifold)
        else:
            obstruction_classes = [None]
        list_obstruction_classes = True
    elif obstruction_class == 'all_generalized':
        obstruction_classes = get_generalized_ptolemy_obstruction_classes(manifold, N)
        list_obstruction_classes = True
    else:
        if N == 2:
            obstruction_classes = get_ptolemy_obstruction_classes(manifold)
        else:
            obstruction_classes = get_generalized_ptolemy_obstruction_classes(manifold, N)
        if obstruction_class == 'all':
            list_obstruction_classes = True
    if list_obstruction_classes:
        return PtolemyVarietyList([PtolemyVariety(manifold, N, obstruction_class, simplify=simplify, eliminate_fixed_ptolemys=eliminate_fixed_ptolemys) for obstruction_class in obstruction_classes])
    try:
        obstruction_class = obstruction_classes[int(obstruction_class)]
    except (KeyError, IndexError):
        raise Exception('Bad index for obstruction class')
    return PtolemyVariety(manifold, N, obstruction_class, simplify=simplify, eliminate_fixed_ptolemys=eliminate_fixed_ptolemys)