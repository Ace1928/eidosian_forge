def get_num_z_categoricals(model_size, override=None):
    if override is not None:
        return override
    assert model_size in _ALLOWED_MODEL_DIMS
    gru_units = {'nano': 4, 'micro': 8, 'mini': 16, 'XXS': 32, 'XS': 32, 'S': 32, 'M': 32, 'L': 32, 'XL': 32}
    return gru_units[model_size]