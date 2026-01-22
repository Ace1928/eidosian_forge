import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union
from .. import mult, orient
from ._font import Font
def render_transform(self) -> List[float]:
    """Effective transform matrix accounting for font size, Tz, and Ts."""
    return mult(self.font_size_matrix(), self.transform)