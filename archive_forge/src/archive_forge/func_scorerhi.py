from .functions import defun, defun_wrapped
@defun
def scorerhi(ctx, z, **kwargs):
    return _scorer(ctx, z, 1, kwargs)