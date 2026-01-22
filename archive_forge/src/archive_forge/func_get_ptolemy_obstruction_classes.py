from . import matrix
from . import homology
from .ptolemyObstructionClass import PtolemyObstructionClass
from .ptolemyGeneralizedObstructionClass import PtolemyGeneralizedObstructionClass
from .ptolemyVariety import PtolemyVariety
from .utilities import MethodMappingList
def get_ptolemy_obstruction_classes(manifold):
    """
    Generates a list of obstruction cocycles representing each class in
    H^2(M,bd M; Z/2) suitable as argument for get_ptolemy_variety.
    The first element in the list is always the trivial obstruction class.

    See Definition 1.7 of
    Garoufalidis, Thurston, Zickert
    The Complex Volume of SL(n,C)-Representations of 3-Manifolds
    http://arxiv.org/abs/1111.2828

    s_f_t takes values +/-1 and is the value of evaluating the cocycle on
    face f of tetrahedron t.

    === Examples ===

    Get the obstruction classes for 4_1:

    >>> from snappy import Manifold
    >>> M = Manifold("4_1")
    >>> c = get_ptolemy_obstruction_classes(M)

    There are two such classes for 4_1:

    >>> len(c)
    2

    Print the non-trivial obstruction class:

    >>> c[1]
    PtolemyObstructionClass(s_0_0 + 1, s_1_0 - 1, s_2_0 - 1, s_3_0 + 1, s_0_0 - s_0_1, s_1_0 - s_3_1, s_2_0 - s_2_1, s_3_0 - s_1_1)

    Construct Ptolemy variety for non-trivial obstruction class:

    >>> p = get_ptolemy_variety(M, N = 2, obstruction_class = c[1])

    Short cut for the above code:

    >>> p = get_ptolemy_variety(M, N = 2, obstruction_class = 1)

    Obstruction class only makes sense for even N:

    >>> p = get_ptolemy_variety(M, N = 3, obstruction_class = c[1])
    Traceback (most recent call last):
        ...
    AssertionError: PtolemyObstructionClass only makes sense for even N, try PtolemyGeneralizedObstructionClass


    When specifying N = 3, it automatically uses generalized obstruction class.

    >>> len(get_ptolemy_variety(M, N = 3, obstruction_class = 'all'))
    2
    """
    H2_elements, explain_columns = get_obstruction_classes(manifold, 2)
    identified_face_classes = manifold._ptolemy_equations_identified_face_classes()
    return [PtolemyObstructionClass(manifold, index, H2_element, explain_columns, identified_face_classes) for index, H2_element in enumerate(H2_elements)]