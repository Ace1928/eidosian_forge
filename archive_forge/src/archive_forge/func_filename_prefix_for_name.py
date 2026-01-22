import itertools
import os
import re
def filename_prefix_for_name(name):
    if os.path.basename(name) != name:
        raise ValueError(f'Should be a dataset name, not a path: {name}')
    return camelcase_to_snakecase(name)