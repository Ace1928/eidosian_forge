import sys
def patch_sys_module():

    def patched_exc_info(fun):

        def pydev_debugger_exc_info():
            type, value, traceback = fun()
            if type == ImportError:
                if traceback and hasattr(traceback, 'tb_next'):
                    return (type, value, traceback.tb_next)
            return (type, value, traceback)
        return pydev_debugger_exc_info
    system_exc_info = sys.exc_info
    sys.exc_info = patched_exc_info(system_exc_info)
    if not hasattr(sys, 'system_exc_info'):
        sys.system_exc_info = system_exc_info