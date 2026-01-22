import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union
from .. import mult, orient
from ._font import Font
def word_tx(self, word: str, TD_offset: float=0.0) -> float:
    """Horizontal text displacement for any word according this text state"""
    return (self.font_size * ((self.font.word_width(word) - TD_offset) / 1000.0) + self.Tc + word.count(' ') * self.Tw) * (self.Tz / 100.0)