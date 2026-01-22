import re
from .utilities import MethodMappingList
from .component import NonZeroDimensionalComponent
from .coordinates import PtolemyCoordinates
from .rur import RUR
from . import processFileBase
from ..pari import pari
def parse_maple_like_rur(text):
    m = re.match('(.*?)\\s*=\\s*0\\s*,\\s*\\{(.*?)\\}', text, re.DOTALL)
    if not m:
        raise Exception('Format not detected')
    poly_text, assignments_text = m.groups()
    var = _find_var_of_poly(poly_text)
    poly = pari(poly_text.replace(var, 'x'))
    assignments_text = assignments_text.replace(var, 'x')
    return dict([_parse_assignment(assignment, poly) for assignment in assignments_text.split(',')])