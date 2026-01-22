def get_gru_units(model_size, override=None):
    if override is not None:
        return override
    assert model_size in _ALLOWED_MODEL_DIMS
    gru_units = {'nano': 16, 'micro': 32, 'mini': 64, 'XXS': 128, 'XS': 256, 'S': 512, 'M': 1024, 'L': 2048, 'XL': 4096}
    return gru_units[model_size]