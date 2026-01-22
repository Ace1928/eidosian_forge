from __future__ import annotations
import os
import sys
import typing as T
from .. import mparser, mesonlib
from .. import environment
from ..interpreterbase import (
from ..interpreter import (
from ..mparser import (
def func_subdir(self, node: BaseNode, args: T.List[TYPE_var], kwargs: T.Dict[str, TYPE_var]) -> None:
    args = self.flatten_args(args)
    if len(args) != 1 or not isinstance(args[0], str):
        sys.stderr.write(f'Unable to evaluate subdir({args}) in AstInterpreter --> Skipping\n')
        return
    prev_subdir = self.subdir
    subdir = os.path.join(prev_subdir, args[0])
    absdir = os.path.join(self.source_root, subdir)
    buildfilename = os.path.join(subdir, environment.build_filename)
    absname = os.path.join(self.source_root, buildfilename)
    symlinkless_dir = os.path.realpath(absdir)
    build_file = os.path.join(symlinkless_dir, 'meson.build')
    if build_file in self.processed_buildfiles:
        sys.stderr.write('Trying to enter {} which has already been visited --> Skipping\n'.format(args[0]))
        return
    self.processed_buildfiles.add(build_file)
    if not os.path.isfile(absname):
        sys.stderr.write(f'Unable to find build file {buildfilename} --> Skipping\n')
        return
    with open(absname, encoding='utf-8') as f:
        code = f.read()
    assert isinstance(code, str)
    try:
        codeblock = mparser.Parser(code, absname).parse()
    except mesonlib.MesonException as me:
        me.file = absname
        raise me
    self.subdir = subdir
    for i in self.visitors:
        codeblock.accept(i)
    self.evaluate_codeblock(codeblock)
    self.subdir = prev_subdir