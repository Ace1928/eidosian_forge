from .component import NonZeroDimensionalComponent
from . import processFileBase
from . import processRurFile
from . import utilities
from . import coordinates
from .polynomial import Polynomial
from .ptolemyVarietyPrimeIdealGroebnerBasis import PtolemyVarietyPrimeIdealGroebnerBasis
def decomposition_from_components(text):
    py_eval = processFileBase.get_py_eval(text)
    manifold_thunk = processFileBase.get_manifold_thunk(text)
    variables_section = processFileBase.find_unique_section(text, 'VARIABLES')
    variables = [remove_optional_quotes(v.strip()) for v in variables_section.split(',') if v.strip()]
    decomposition = processFileBase.find_unique_section(text, 'IDEAL=COMPONENTS')
    params, body = processFileBase.extract_parameters_and_body_from_section(decomposition)
    if 'TYPE' not in params.keys():
        raise Exception('No TYPE given for IDEAL=COMPONENTS')
    type = params['TYPE'].strip()
    if not type == 'PTOLEMY':
        raise Exception("TYPE '%s' not supported" % type)
    components = processFileBase.find_section(text, 'COMPONENT')
    return utilities.MethodMappingList([process_component(py_eval, manifold_thunk, variables, component) for component in components])