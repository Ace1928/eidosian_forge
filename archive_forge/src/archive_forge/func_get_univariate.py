import scipy.stats as stats
from ..exceptions import PlotnineError
def get_univariate(name):
    """
    Get univariate scipy.stats distribution of a given name
    """
    if name not in univariate:
        msg = "Unknown univariate distribution '{}'"
        raise ValueError(msg.format(name))
    return get(name)