from __future__ import annotations
import typing
def get_palette_module(scheme: ColorScheme | ColorSchemeShort) -> ModuleType:
    """
    Return Module with the palettes for the scheme
    """
    if scheme in ('sequential', 'seq'):
        from . import sequential
        return sequential
    elif scheme in ('qualitative', 'qual'):
        from . import qualitative
        return qualitative
    elif scheme in ('diverging', 'div'):
        from . import diverging
        return diverging
    else:
        raise ValueError(f'Unknown type of brewer palette: {type}')