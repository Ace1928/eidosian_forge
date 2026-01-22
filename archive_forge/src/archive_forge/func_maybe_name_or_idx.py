from statsmodels.compat.python import lrange
def maybe_name_or_idx(idx, model):
    """
    Give a name or an integer and return the name and integer location of the
    column in a design matrix.
    """
    if idx is None:
        idx = lrange(model.exog.shape[1])
    if isinstance(idx, int):
        exog_name = model.exog_names[idx]
        exog_idx = idx
    elif isinstance(idx, (tuple, list)):
        exog_name = []
        exog_idx = []
        for item in idx:
            exog_name_item, exog_idx_item = maybe_name_or_idx(item, model)
            exog_name.append(exog_name_item)
            exog_idx.append(exog_idx_item)
    else:
        exog_name = idx
        exog_idx = model.exog_names.index(idx)
    return (exog_name, exog_idx)