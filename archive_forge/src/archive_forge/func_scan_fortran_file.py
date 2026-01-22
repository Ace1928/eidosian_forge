from __future__ import annotations
import json
import os
import pathlib
import pickle
import re
import sys
import typing as T
from ..backend.ninjabackend import ninja_quote
from ..compilers.compilers import lang_suffixes
def scan_fortran_file(self, fname: str) -> None:
    fpath = pathlib.Path(fname)
    modules_in_this_file = set()
    for line in fpath.read_text(encoding='utf-8', errors='ignore').split('\n'):
        import_match = FORTRAN_USE_RE.match(line)
        export_match = FORTRAN_MODULE_RE.match(line)
        submodule_export_match = FORTRAN_SUBMOD_RE.match(line)
        if import_match:
            needed = import_match.group(1).lower()
            if needed not in modules_in_this_file:
                if fname in self.needs:
                    self.needs[fname].append(needed)
                else:
                    self.needs[fname] = [needed]
        if export_match:
            exported_module = export_match.group(1).lower()
            assert exported_module not in modules_in_this_file
            modules_in_this_file.add(exported_module)
            if exported_module in self.provided_by:
                raise RuntimeError(f'Multiple files provide module {exported_module}.')
            self.sources_with_exports.append(fname)
            self.provided_by[exported_module] = fname
            self.exports[fname] = exported_module
        if submodule_export_match:
            parent_module_name_full = submodule_export_match.group(1).lower()
            parent_module_name = parent_module_name_full.split(':')[0]
            submodule_name = submodule_export_match.group(2).lower()
            concat_name = f'{parent_module_name}:{submodule_name}'
            self.sources_with_exports.append(fname)
            self.provided_by[concat_name] = fname
            self.exports[fname] = concat_name
            if fname in self.needs:
                self.needs[fname].append(parent_module_name_full)
            else:
                self.needs[fname] = [parent_module_name_full]