import pytest
from ..tractogram import Tractogram
from ..tractogram_file import TractogramFile
@classmethod
def is_correct_format(cls, fileobj):
    return False