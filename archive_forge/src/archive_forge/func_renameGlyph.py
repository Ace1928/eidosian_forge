from __future__ import annotations
from typing import (
from attrs import define, field
from fontTools.ufoLib.glifLib import GlyphSet
from ufoLib2.constants import DEFAULT_LAYER_NAME
from ufoLib2.objects.glyph import Glyph
from ufoLib2.objects.lib import (
from ufoLib2.objects.misc import (
from ufoLib2.serde import serde
from ufoLib2.typing import T
def renameGlyph(self, name: str, newName: str, overwrite: bool=False) -> None:
    """Renames a Glyph object in this layer.

        Args:
            name: The old name.
            newName: The new name.
            overwrite: If False, raises exception if newName is already taken.
                If True, overwrites (read: deletes) the old Glyph object.
        """
    if name == newName:
        return
    if not overwrite and newName in self._glyphs:
        raise KeyError(f"target glyph named '{newName}' already exists")
    glyph = self.pop(name)
    glyph._name = newName
    self._glyphs[newName] = glyph