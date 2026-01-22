import scipy.stats as stats
from ..exceptions import PlotnineError
def get_continuous_distribution(name):
    """
    Get continuous scipy.stats distribution of a given name
    """
    if name not in continuous:
        msg = "Unknown continuous distribution '{}'"
        raise ValueError(msg.format(name))
    return getattr(stats, name)