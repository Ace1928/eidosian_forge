import itertools
import os
import re
def filepattern_for_dataset_split(dataset_name, split, data_dir, filetype_suffix=None):
    prefix = filename_prefix_for_split(dataset_name, split)
    if filetype_suffix:
        prefix += f'.{filetype_suffix}'
    filepath = os.path.join(data_dir, prefix)
    return f'{filepath}*'