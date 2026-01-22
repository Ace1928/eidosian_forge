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
@defaultLayer.setter
def defaultLayer(self, layer: Layer) -> None:
    if layer is self._defaultLayer:
        return
    if layer not in self._layers.values():
        raise ValueError(f"Layer {layer!r} not found in layer set; can't set as default")
    if self._defaultLayer.name == DEFAULT_LAYER_NAME:
        raise ValueError("there's already a layer named 'public.default' which must stay default")
    self._defaultLayer._default = False
    layer._default = True
    self._defaultLayer = layer