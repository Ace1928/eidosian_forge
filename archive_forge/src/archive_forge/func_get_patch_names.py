import os
import re
from typing import Iterator, List, Optional
from .errors import BzrError
def get_patch_names(iter_lines):
    line = next(iter_lines)
    try:
        match = re.match(binary_files_re, line)
        if match is not None:
            raise BinaryFiles(match.group(1), match.group(2))
        if not line.startswith(b'--- '):
            raise MalformedPatchHeader('No orig name', line)
        else:
            orig_name = line[4:].rstrip(b'\n')
            try:
                orig_name, orig_ts = orig_name.split(b'\t')
            except ValueError:
                orig_ts = None
    except StopIteration:
        raise MalformedPatchHeader('No orig line', '')
    try:
        line = next(iter_lines)
        if not line.startswith(b'+++ '):
            raise PatchSyntax('No mod name')
        else:
            mod_name = line[4:].rstrip(b'\n')
            try:
                mod_name, mod_ts = mod_name.split(b'\t')
            except ValueError:
                mod_ts = None
    except StopIteration:
        raise MalformedPatchHeader('No mod line', '')
    return ((orig_name, orig_ts), (mod_name, mod_ts))