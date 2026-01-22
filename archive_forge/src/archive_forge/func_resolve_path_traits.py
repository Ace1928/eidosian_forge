from collections.abc import Sequence
from traits import __version__ as traits_version
import traits.api as traits
from traits.api import TraitType, Unicode
from traits.trait_base import _Undefined
from pathlib import Path
from ...utils.filemanip import path_resolve
def resolve_path_traits(thistrait, value, cwd):
    """Resolve a BasePath-derived trait given an interface spec."""
    return _recurse_on_path_traits(_resolve_path, thistrait, value, cwd)