from __future__ import annotations
from typing import (
from attrs import define, field
from fontTools.ufoLib import UFOReader, UFOWriter
from ufoLib2.constants import DEFAULT_LAYER_NAME
from ufoLib2.errors import Error
from ufoLib2.objects.layer import Layer
from ufoLib2.objects.misc import (
from ufoLib2.serde import serde
from ufoLib2.typing import T
def loadLayer(self, layerName: str, lazy: bool=True) -> Layer:
    assert self._reader is not None
    if layerName not in self._layers:
        raise KeyError(layerName)
    layer = self._loadLayer(self._reader, layerName, lazy)
    self._layers[layerName] = layer
    return layer