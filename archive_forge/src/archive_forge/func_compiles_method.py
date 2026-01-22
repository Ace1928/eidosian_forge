from __future__ import annotations
import collections
import enum
import functools
import os
import itertools
import typing as T
from .. import build
from .. import coredata
from .. import dependencies
from .. import mesonlib
from .. import mlog
from ..compilers import SUFFIX_TO_LANG
from ..compilers.compilers import CompileCheckMode
from ..interpreterbase import (ObjectHolder, noPosargs, noKwargs,
from ..interpreterbase.decorators import ContainerTypeInfo, typed_kwargs, KwargInfo, typed_pos_args
from ..mesonlib import OptionKey
from .interpreterobjects import (extract_required_kwarg, extract_search_dirs)
from .type_checking import REQUIRED_KW, in_set_validator, NoneType
@typed_pos_args('compiler.compiles', (str, mesonlib.File))
@typed_kwargs('compiler.compiles', *_COMPILES_KWS)
def compiles_method(self, args: T.Tuple['mesonlib.FileOrString'], kwargs: 'CompileKW') -> bool:
    code = args[0]
    if isinstance(code, mesonlib.File):
        if code.is_built:
            FeatureNew.single_use('compiler.compiles with file created at setup time', '1.2.0', self.subproject, 'It was broken and either errored or returned false.', self.current_node)
        self.interpreter.add_build_def_file(code)
        code = mesonlib.File.from_absolute_file(code.absolute_path(self.environment.source_dir, self.environment.build_dir))
    testname = kwargs['name']
    extra_args = functools.partial(self._determine_args, kwargs)
    deps, msg = self._determine_dependencies(kwargs['dependencies'], endl=None)
    result, cached = self.compiler.compiles(code, self.environment, extra_args=extra_args, dependencies=deps)
    if testname:
        if result:
            h = mlog.green('YES')
        else:
            h = mlog.red('NO')
        cached_msg = mlog.blue('(cached)') if cached else ''
        mlog.log('Checking if', mlog.bold(testname, True), msg, 'compiles:', h, cached_msg)
    return result