from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Mapping, Tuple
from ufoLib2.serde import serde
@classmethod
def from_nested_dicts(self, kerning: Mapping[str, Mapping[str, float]]) -> Kerning:
    return Kerning((((left, right), kerning[left][right]) for left in kerning for right in kerning[left]))