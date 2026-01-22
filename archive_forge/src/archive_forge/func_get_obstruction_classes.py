from . import matrix
from . import homology
from .ptolemyObstructionClass import PtolemyObstructionClass
from .ptolemyGeneralizedObstructionClass import PtolemyGeneralizedObstructionClass
from .ptolemyVariety import PtolemyVariety
from .utilities import MethodMappingList
def get_obstruction_classes(manifold, N):
    chain_d3, dummy_rows, dummy_columns = manifold._ptolemy_equations_boundary_map_3()
    chain_d2, dummy_rows, explain_columns = manifold._ptolemy_equations_boundary_map_2()
    cochain_d2 = matrix.matrix_transpose(chain_d3)
    cochain_d1 = matrix.matrix_transpose(chain_d2)
    return (homology.homology_representatives(cochain_d2, cochain_d1, N), explain_columns)