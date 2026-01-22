import pytest
from .common import TestCase
from h5py import File
def throwing(name, obj):
    print(name, obj)
    raise SampleException('throwing exception')