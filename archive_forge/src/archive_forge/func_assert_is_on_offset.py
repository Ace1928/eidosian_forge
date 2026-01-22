from __future__ import annotations
def assert_is_on_offset(offset, date, expected):
    actual = offset.is_on_offset(date)
    assert actual == expected, f'\nExpected: {expected}\nActual: {actual}\nFor Offset: {offset})\nAt Date: {date}'