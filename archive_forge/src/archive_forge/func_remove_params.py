from lib2to3 import fixer_base
from lib2to3.fixer_util import token, String, Newline, Comma, Name
from libfuturize.fixer_util import indentation, suitify, DoubleStar
def remove_params(raw_params, kwargs_default=_kwargs_default_name):
    u"""
    Removes all keyword-only args from the params list and a bare star, if any.
    Does not add the kwargs dict if needed.
    Returns True if more action is needed, False if not
    (more action is needed if no kwargs dict exists)
    """
    assert raw_params[0].type == token.STAR
    if raw_params[1].type == token.COMMA:
        raw_params[0].remove()
        raw_params[1].remove()
        kw_params = raw_params[2:]
    else:
        kw_params = raw_params[3:]
    for param in kw_params:
        if param.type != token.DOUBLESTAR:
            param.remove()
        else:
            return False
    else:
        return True