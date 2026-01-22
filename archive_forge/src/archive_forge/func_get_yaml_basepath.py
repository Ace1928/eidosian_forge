import logging
import os
def get_yaml_basepath():
    """Returns the full path of the directory in which builtins are located."""
    if _handler_dir is None:
        set_builtins_dir(DEFAULT_DIR)
    return _handler_dir