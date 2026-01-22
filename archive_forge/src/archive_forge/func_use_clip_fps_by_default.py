import decorator
from moviepy.tools import cvsecs
@decorator.decorator
def use_clip_fps_by_default(f, clip, *a, **k):
    """ Will use clip.fps if no fps=... is provided in **k """

    def fun(fps):
        if fps is not None:
            return fps
        elif getattr(clip, 'fps', None):
            return clip.fps
        raise AttributeError("No 'fps' (frames per second) attribute specified for function %s and the clip has no 'fps' attribute. Either provide e.g. fps=24 in the arguments of the function, or define the clip's fps with `clip.fps=24`" % f.__name__)
    if hasattr(f, 'func_code'):
        func_code = f.func_code
    else:
        func_code = f.__code__
    names = func_code.co_varnames[1:]
    new_a = [fun(arg) if name == 'fps' else arg for arg, name in zip(a, names)]
    new_kw = {k: fun(v) if k == 'fps' else v for k, v in k.items()}
    return f(clip, *new_a, **new_kw)