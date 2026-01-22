from .functions import defun, defun_wrapped
def wpzeros(t):
    """Precision needed to compute higher zeros"""
    wp = 53
    if t > 3 * 10 ** 8:
        wp = 63
    if t > 10 ** 11:
        wp = 70
    if t > 10 ** 14:
        wp = 83
    return wp