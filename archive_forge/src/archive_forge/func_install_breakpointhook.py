import sys
def install_breakpointhook():

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
    if sys.version_info[0:2] >= (3, 7):
        sys.breakpointhook = custom_sitecustomize_breakpointhook
    else:
        if sys.version_info[0] >= 3:
            import builtins as __builtin__
        else:
            import __builtin__
        __builtin__.breakpoint = custom_sitecustomize_breakpointhook
        sys.__breakpointhook__ = custom_sitecustomize_breakpointhook