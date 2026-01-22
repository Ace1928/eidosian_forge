import sys
def pydev_debugger_reload(module):
    orig_reload(module)
    if module.__name__ == 'sys':
        patch_sys_module()