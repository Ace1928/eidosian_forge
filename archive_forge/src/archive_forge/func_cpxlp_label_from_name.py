import re
from pyomo.common.deprecation import deprecated
from pyomo.core.base.componentuid import ComponentUID
def cpxlp_label_from_name(name):
    if name is None:
        raise RuntimeError('Illegal name=None supplied to cpxlp_label_from_name function')
    return str.translate(name, _cpxlp_translation_table)