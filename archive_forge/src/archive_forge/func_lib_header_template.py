from __future__ import annotations
import abc
import re
import typing as T
@abc.abstractproperty
def lib_header_template(self) -> str:
    pass