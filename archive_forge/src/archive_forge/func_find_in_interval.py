from .functions import defun, defun_wrapped
def find_in_interval(ctx, f, ab):
    return ctx.findroot(f, ab, solver='illinois', verify=False)