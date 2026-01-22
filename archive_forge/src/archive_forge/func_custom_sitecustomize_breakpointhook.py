import sys
def custom_sitecustomize_breakpointhook(*args, **kwargs):
    import os
    hookname = os.getenv('PYTHONBREAKPOINT')
    if hookname is not None and len(hookname) > 0 and hasattr(sys, '__breakpointhook__') and (sys.__breakpointhook__ != custom_sitecustomize_breakpointhook):
        sys.__breakpointhook__(*args, **kwargs)
    else:
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        import pydevd
        kwargs.setdefault('stop_at_frame', sys._getframe().f_back)
        pydevd.settrace(*args, **kwargs)