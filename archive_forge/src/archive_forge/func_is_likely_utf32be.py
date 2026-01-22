from typing import List, Union
from .charsetprober import CharSetProber
from .enums import ProbingState
def is_likely_utf32be(self) -> bool:
    approx_chars = self.approx_32bit_chars()
    return approx_chars >= self.MIN_CHARS_FOR_DETECTION and (self.zeros_at_mod[0] / approx_chars > self.EXPECTED_RATIO and self.zeros_at_mod[1] / approx_chars > self.EXPECTED_RATIO and (self.zeros_at_mod[2] / approx_chars > self.EXPECTED_RATIO) and (self.nonzeros_at_mod[3] / approx_chars > self.EXPECTED_RATIO) and (not self.invalid_utf32be))