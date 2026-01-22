import re
from .utilities import MethodMappingList
from .component import NonZeroDimensionalComponent
from .coordinates import PtolemyCoordinates
from .rur import RUR
from . import processFileBase
from ..pari import pari
def decomposition_from_rur(text):
    py_eval = processFileBase.get_py_eval(text)
    manifold_thunk = processFileBase.get_manifold_thunk(text)
    rursection = processFileBase.find_unique_section(text, 'RUR=DECOMPOSITION')
    rurs = processFileBase.find_section(rursection, 'COMPONENT')
    result = MethodMappingList([SolutionContainer(_process_rur_component(rur, py_eval, manifold_thunk)) for rur in rurs])
    return result