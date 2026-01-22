from functools import lru_cache
from traits.observation import _generated_parser
import traits.observation.expression as expression_module
@lru_cache(maxsize=expression_module._OBSERVER_EXPRESSION_CACHE_MAXSIZE)
def compile_str(text):
    """ Compile a mini-language string to a list of ObserverGraphs.

    Parameters
    ----------
    text : str
        Text to be parsed.

    Returns
    -------
    list of ObserverGraph
    """
    return expression_module.compile_expr(parse(text))