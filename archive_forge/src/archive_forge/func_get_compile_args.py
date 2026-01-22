from __future__ import annotations
from ..mesonlib import MesonException, OptionKey
from .. import mlog
from pathlib import Path
import typing as T
def get_compile_args(self, tgt: str, lang: str, initial: T.List[str]) -> T.List[str]:
    initial = self.global_options.get_compile_args(lang, initial)
    if tgt in self.target_options:
        initial = self.target_options[tgt].get_compile_args(lang, initial)
    return initial