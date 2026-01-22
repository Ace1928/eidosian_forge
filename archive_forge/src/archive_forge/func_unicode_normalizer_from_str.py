from .. import normalizers
def unicode_normalizer_from_str(normalizer: str) -> Normalizer:
    if normalizer not in NORMALIZERS:
        raise ValueError('{} is not a known unicode normalizer. Available are {}'.format(normalizer, NORMALIZERS.keys()))
    return NORMALIZERS[normalizer]()