import array
from typing import (
def output_line(self, y: int, bits: Sequence[int]) -> None:
    for x, b in enumerate(bits):
        if b:
            self.img.set_at((x, y), (255, 255, 255))
        else:
            self.img.set_at((x, y), (0, 0, 0))
    return