from collections import OrderedDict
def check_serializing_named_tensor(tensor):
    if tensor.has_names():
        raise RuntimeError("NYI: Named tensors don't support serialization. Please drop names via `tensor = tensor.rename(None)` before serialization.")