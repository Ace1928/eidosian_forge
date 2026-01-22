def merge_clouds(old_dict, new_dict):
    """Like dict.update, except handling nested dicts."""
    ret = old_dict.copy()
    for k, v in new_dict.items():
        if isinstance(v, dict):
            if k in ret:
                ret[k] = merge_clouds(ret[k], v)
            else:
                ret[k] = v.copy()
        else:
            ret[k] = v
    return ret