import sys
from math import trunc
from typing import (
def describe_multi(self, timeframes: Sequence[Tuple[TimeFrameLiteral, Union[int, float]]], only_distance: bool=False) -> str:
    """Describes a delta within multiple timeframes in plain language.
        In Hebrew, the and word behaves a bit differently.

        :param timeframes: a list of string, quantity pairs each representing a timeframe and delta.
        :param only_distance: return only distance eg: "2 hours and 11 seconds" without "in" or "ago" keywords
        """
    humanized = ''
    for index, (timeframe, delta) in enumerate(timeframes):
        last_humanized = self._format_timeframe(timeframe, trunc(delta))
        if index == 0:
            humanized = last_humanized
        elif index == len(timeframes) - 1:
            humanized += ' ' + self.and_word
            if last_humanized[0].isdecimal():
                humanized += '־'
            humanized += last_humanized
        else:
            humanized += ', ' + last_humanized
    if not only_distance:
        humanized = self._format_relative(humanized, timeframe, trunc(delta))
    return humanized