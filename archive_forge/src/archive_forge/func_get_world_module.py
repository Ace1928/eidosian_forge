import importlib
import math
import re
from enum import Enum
def get_world_module(world_path):
    """
    Import the module specified by the world_path.
    """
    run_module = None
    try:
        run_module = importlib.import_module(world_path)
    except Exception as e:
        print('Could not import world file {}'.format(world_path))
        raise e
    return run_module