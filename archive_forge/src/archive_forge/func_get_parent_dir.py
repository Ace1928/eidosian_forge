import os
import importlib
def get_parent_dir(module):
    return os.path.abspath(os.path.join(os.path.dirname(module.__file__), '..'))