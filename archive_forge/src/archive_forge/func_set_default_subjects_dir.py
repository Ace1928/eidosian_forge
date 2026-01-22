import os
from ... import LooseVersion
from ...utils.filemanip import fname_presuffix
from ..base import (
@classmethod
def set_default_subjects_dir(cls, subjects_dir):
    cls._subjects_dir = subjects_dir