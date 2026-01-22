from OpenGL.EGL import *
import itertools
def debug_configs(display, configs=None, max_count=256):
    """Present a formatted list of configs for the display"""
    if configs is None:
        configs = (EGLConfig * max_count)()
        num_configs = EGLint()
        eglGetConfigs(display, configs, max_count, num_configs)
        if not num_configs.value:
            return []
        configs = configs[:num_configs.value]
    debug_configs = [debug_config(display, cfg) for cfg in configs]
    return debug_configs