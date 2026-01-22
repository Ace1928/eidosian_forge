import pytest
from string import ascii_letters
from random import randint
import gc
import sys
def test_parse_kv(kivy_benchmark):
    from kivy.lang import Builder
    suffix = 0

    def parse_kv():
        nonlocal suffix
        Builder.load_string(f'\n<MyWidget{suffix}>:\n    width: 55\n    height: 37\n    x: self.width + 5\n    y: self.height + 32\n')
        suffix += 1
    parse_kv()
    kivy_benchmark(parse_kv)