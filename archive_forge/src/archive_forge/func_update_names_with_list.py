from collections import OrderedDict
def update_names_with_list(tensor, names, inplace):
    if len(names) == 1 and names[0] is None:
        return tensor._update_names(None, inplace)
    return tensor._update_names(resolve_ellipsis(names, tensor.names, namer_api_name(inplace)), inplace)