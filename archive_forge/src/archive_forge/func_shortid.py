import email.parser
import time
from difflib import SequenceMatcher
from typing import BinaryIO, Optional, TextIO, Union
from .objects import S_ISGITLINK, Blob, Commit
from .pack import ObjectContainer
def shortid(hexsha):
    if hexsha is None:
        return b'0' * 7
    else:
        return hexsha[:7]