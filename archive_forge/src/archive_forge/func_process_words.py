import re
import unicodedata
from fractions import Fraction
from typing import Iterator, List, Match, Optional, Union
import regex
def process_words(self, words: List[str]) -> Iterator[str]:
    prefix: Optional[str] = None
    value: Optional[Union[str, int]] = None
    skip = False

    def to_fraction(s: str):
        try:
            return Fraction(s)
        except ValueError:
            return None

    def output(result: Union[str, int]):
        nonlocal prefix, value
        result = str(result)
        if prefix is not None:
            result = prefix + result
        value = None
        prefix = None
        return result
    if len(words) == 0:
        return
    for i, current in enumerate(words):
        prev = words[i - 1] if i != 0 else None
        next = words[i + 1] if i != len(words) - 1 else None
        if skip:
            skip = False
            continue
        next_is_numeric = next is not None and re.match('^\\d+(\\.\\d+)?$', next)
        has_prefix = current[0] in self.prefixes
        current_without_prefix = current[1:] if has_prefix else current
        if re.match('^\\d+(\\.\\d+)?$', current_without_prefix):
            f = to_fraction(current_without_prefix)
            if f is None:
                raise ValueError('Converting the fraction failed')
            if value is not None:
                if isinstance(value, str) and value.endswith('.'):
                    value = str(value) + str(current)
                    continue
                else:
                    yield output(value)
            prefix = current[0] if has_prefix else prefix
            if f.denominator == 1:
                value = f.numerator
            else:
                value = current_without_prefix
        elif current not in self.words:
            if value is not None:
                yield output(value)
            yield output(current)
        elif current in self.zeros:
            value = str(value or '') + '0'
        elif current in self.ones:
            ones = self.ones[current]
            if value is None:
                value = ones
            elif isinstance(value, str) or prev in self.ones:
                if prev in self.tens and ones < 10:
                    value = value[:-1] + str(ones)
                else:
                    value = str(value) + str(ones)
            elif ones < 10:
                if value % 10 == 0:
                    value += ones
                else:
                    value = str(value) + str(ones)
            elif value % 100 == 0:
                value += ones
            else:
                value = str(value) + str(ones)
        elif current in self.ones_suffixed:
            ones, suffix = self.ones_suffixed[current]
            if value is None:
                yield output(str(ones) + suffix)
            elif isinstance(value, str) or prev in self.ones:
                if prev in self.tens and ones < 10:
                    yield output(value[:-1] + str(ones) + suffix)
                else:
                    yield output(str(value) + str(ones) + suffix)
            elif ones < 10:
                if value % 10 == 0:
                    yield output(str(value + ones) + suffix)
                else:
                    yield output(str(value) + str(ones) + suffix)
            elif value % 100 == 0:
                yield output(str(value + ones) + suffix)
            else:
                yield output(str(value) + str(ones) + suffix)
            value = None
        elif current in self.tens:
            tens = self.tens[current]
            if value is None:
                value = tens
            elif isinstance(value, str):
                value = str(value) + str(tens)
            elif value % 100 == 0:
                value += tens
            else:
                value = str(value) + str(tens)
        elif current in self.tens_suffixed:
            tens, suffix = self.tens_suffixed[current]
            if value is None:
                yield output(str(tens) + suffix)
            elif isinstance(value, str):
                yield output(str(value) + str(tens) + suffix)
            elif value % 100 == 0:
                yield output(str(value + tens) + suffix)
            else:
                yield output(str(value) + str(tens) + suffix)
        elif current in self.multipliers:
            multiplier = self.multipliers[current]
            if value is None:
                value = multiplier
            elif isinstance(value, str) or value == 0:
                f = to_fraction(value)
                p = f * multiplier if f is not None else None
                if f is not None and p.denominator == 1:
                    value = p.numerator
                else:
                    yield output(value)
                    value = multiplier
            else:
                before = value // 1000 * 1000
                residual = value % 1000
                value = before + residual * multiplier
        elif current in self.multipliers_suffixed:
            multiplier, suffix = self.multipliers_suffixed[current]
            if value is None:
                yield output(str(multiplier) + suffix)
            elif isinstance(value, str):
                f = to_fraction(value)
                p = f * multiplier if f is not None else None
                if f is not None and p.denominator == 1:
                    yield output(str(p.numerator) + suffix)
                else:
                    yield output(value)
                    yield output(str(multiplier) + suffix)
            else:
                before = value // 1000 * 1000
                residual = value % 1000
                value = before + residual * multiplier
                yield output(str(value) + suffix)
            value = None
        elif current in self.preceding_prefixers:
            if value is not None:
                yield output(value)
            if next in self.words or next_is_numeric:
                prefix = self.preceding_prefixers[current]
            else:
                yield output(current)
        elif current in self.following_prefixers:
            if value is not None:
                prefix = self.following_prefixers[current]
                yield output(value)
            else:
                yield output(current)
        elif current in self.suffixers:
            if value is not None:
                suffix = self.suffixers[current]
                if isinstance(suffix, dict):
                    if next in suffix:
                        yield output(str(value) + suffix[next])
                        skip = True
                    else:
                        yield output(value)
                        yield output(current)
                else:
                    yield output(str(value) + suffix)
            else:
                yield output(current)
        elif current in self.specials:
            if next not in self.words and (not next_is_numeric):
                if value is not None:
                    yield output(value)
                yield output(current)
            elif current == 'and':
                if prev not in self.multipliers:
                    if value is not None:
                        yield output(value)
                    yield output(current)
            elif current == 'double' or current == 'triple':
                if next in self.ones or next in self.zeros:
                    repeats = 2 if current == 'double' else 3
                    ones = self.ones.get(next, 0)
                    value = str(value or '') + str(ones) * repeats
                    skip = True
                else:
                    if value is not None:
                        yield output(value)
                    yield output(current)
            elif current == 'point':
                if next in self.decimals or next_is_numeric:
                    value = str(value or '') + '.'
            else:
                raise ValueError(f'Unexpected token: {current}')
        else:
            raise ValueError(f'Unexpected token: {current}')
    if value is not None:
        yield output(value)