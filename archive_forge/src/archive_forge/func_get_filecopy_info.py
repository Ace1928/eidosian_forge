import os
from inspect import isclass
from copy import deepcopy
from warnings import warn
from packaging.version import Version
from traits.trait_errors import TraitError
from traits.trait_handlers import TraitDictObject, TraitListObject
from ...utils.filemanip import md5, hash_infile, hash_timestamp
from .traits_extension import (
from ... import config, __version__
def get_filecopy_info(cls):
    """Provides information about file inputs to copy or link to cwd.
    Necessary for pipeline operation
    """
    if cls.input_spec is None:
        return None
    if not isclass(cls) and hasattr(cls, 'normalize_filenames'):
        cls.normalize_filenames()
    info = []
    inputs = cls.input_spec() if isclass(cls) else cls.inputs
    metadata = dict(copyfile=lambda t: t is not None)
    for name, spec in sorted(inputs.traits(**metadata).items()):
        info.append(dict(key=name, copy=spec.copyfile))
    return info