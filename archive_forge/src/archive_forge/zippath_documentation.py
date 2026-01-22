from __future__ import annotations
import errno
import os
import time
from typing import (
from zipfile import ZipFile
from zope.interface import implementer
from typing_extensions import Literal, Self
from twisted.python.compat import cmp, comparable
from twisted.python.filepath import (

        Return the archive file's status change time.
        