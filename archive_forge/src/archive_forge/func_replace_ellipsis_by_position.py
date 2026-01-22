from collections import OrderedDict
def replace_ellipsis_by_position(ellipsis_idx, names, tensor_names):
    globbed_names = expand_single_ellipsis(ellipsis_idx, len(names) - ellipsis_idx - 1, tensor_names)
    return names[:ellipsis_idx] + globbed_names + names[ellipsis_idx + 1:]