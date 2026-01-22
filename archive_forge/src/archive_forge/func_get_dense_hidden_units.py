def get_dense_hidden_units(model_size, override=None):
    if override is not None:
        return override
    assert model_size in _ALLOWED_MODEL_DIMS
    dense_units = {'nano': 16, 'micro': 32, 'mini': 64, 'XXS': 128, 'XS': 256, 'S': 512, 'M': 640, 'L': 768, 'XL': 1024}
    return dense_units[model_size]