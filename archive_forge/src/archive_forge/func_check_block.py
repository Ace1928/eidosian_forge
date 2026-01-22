import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def check_block(self, block, asm):
    self.check_descr(self.descr(block), asm)