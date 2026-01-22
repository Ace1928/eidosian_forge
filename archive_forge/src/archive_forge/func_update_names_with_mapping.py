from collections import OrderedDict
def update_names_with_mapping(tensor, rename_map, inplace):
    dim_map = build_dim_map(tensor)
    for old_dim in rename_map.keys():
        new_dim = rename_map[old_dim]
        if old_dim in dim_map.keys():
            dim_map[old_dim] = new_dim
        else:
            raise RuntimeError(f"{namer_api_name(inplace)}: Tried to rename dim '{old_dim}' to dim {new_dim} in Tensor[{tensor.names}] but dim '{old_dim}' does not exist")
    return tensor._update_names(tuple(dim_map.values()), inplace)