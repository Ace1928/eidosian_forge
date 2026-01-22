import itertools
import os
import re
def filename_prefix_for_split(name, split):
    if os.path.basename(name) != name:
        raise ValueError(f'Should be a dataset name, not a path: {name}')
    if not re.match(_split_re, split):
        raise ValueError(f"Split name should match '{_split_re}'' but got '{split}'.")
    return f'{filename_prefix_for_name(name)}-{split}'