from rpy2 import rinterface
from warnings import warn
from collections import defaultdict
def get_packagepath(package: str) -> str:
    """ return the path to an R package installed """
    res = _find_package(rinterface.StrSexpVector((package,)))
    return res[0]