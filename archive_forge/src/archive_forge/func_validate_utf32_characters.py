from typing import List, Union
from .charsetprober import CharSetProber
from .enums import ProbingState
def validate_utf32_characters(self, quad: List[int]) -> None:
    """
        Validate if the quad of bytes is valid UTF-32.

        UTF-32 is valid in the range 0x00000000 - 0x0010FFFF
        excluding 0x0000D800 - 0x0000DFFF

        https://en.wikipedia.org/wiki/UTF-32
        """
    if quad[0] != 0 or quad[1] > 16 or (quad[0] == 0 and quad[1] == 0 and (216 <= quad[2] <= 223)):
        self.invalid_utf32be = True
    if quad[3] != 0 or quad[2] > 16 or (quad[3] == 0 and quad[2] == 0 and (216 <= quad[1] <= 223)):
        self.invalid_utf32le = True