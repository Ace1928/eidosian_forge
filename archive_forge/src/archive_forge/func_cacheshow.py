import dataclasses
import json
import os
from pathlib import Path
from typing import Dict
from typing import final
from typing import Generator
from typing import Iterable
from typing import List
from typing import Optional
from typing import Set
from typing import Union
from .pathlib import resolve_from_str
from .pathlib import rm_rf
from .reports import CollectReport
from _pytest import nodes
from _pytest._io import TerminalWriter
from _pytest.config import Config
from _pytest.config import ExitCode
from _pytest.config import hookimpl
from _pytest.config.argparsing import Parser
from _pytest.deprecated import check_ispytest
from _pytest.fixtures import fixture
from _pytest.fixtures import FixtureRequest
from _pytest.main import Session
from _pytest.nodes import Directory
from _pytest.nodes import File
from _pytest.reports import TestReport
def cacheshow(config: Config, session: Session) -> int:
    from pprint import pformat
    assert config.cache is not None
    tw = TerminalWriter()
    tw.line('cachedir: ' + str(config.cache._cachedir))
    if not config.cache._cachedir.is_dir():
        tw.line('cache is empty')
        return 0
    glob = config.option.cacheshow[0]
    if glob is None:
        glob = '*'
    dummy = object()
    basedir = config.cache._cachedir
    vdir = basedir / Cache._CACHE_PREFIX_VALUES
    tw.sep('-', 'cache values for %r' % glob)
    for valpath in sorted((x for x in vdir.rglob(glob) if x.is_file())):
        key = str(valpath.relative_to(vdir))
        val = config.cache.get(key, dummy)
        if val is dummy:
            tw.line('%s contains unreadable content, will be ignored' % key)
        else:
            tw.line('%s contains:' % key)
            for line in pformat(val).splitlines():
                tw.line('  ' + line)
    ddir = basedir / Cache._CACHE_PREFIX_DIRS
    if ddir.is_dir():
        contents = sorted(ddir.rglob(glob))
        tw.sep('-', 'cache directories for %r' % glob)
        for p in contents:
            if p.is_file():
                key = str(p.relative_to(basedir))
                tw.line(f'{key} is a file of length {p.stat().st_size:d}')
    return 0