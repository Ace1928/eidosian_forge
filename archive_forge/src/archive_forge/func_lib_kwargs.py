from __future__ import annotations
import abc
import re
import typing as T
def lib_kwargs(self) -> T.Dict[str, str]:
    kwargs = super().lib_kwargs()
    kwargs['header_file'] = f'{self.lowercase_token}.{self.header_ext}'
    return kwargs