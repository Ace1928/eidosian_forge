import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Math.Primality import (
def ending_with_one(number):
    return number % 10 == 1