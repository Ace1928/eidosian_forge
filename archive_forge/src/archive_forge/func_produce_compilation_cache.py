from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import dom
from . import emulation
from . import io
from . import network
from . import runtime
def produce_compilation_cache(scripts: typing.List[CompilationCacheParams]) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Requests backend to produce compilation cache for the specified scripts.
    ``scripts`` are appeneded to the list of scripts for which the cache
    would be produced. The list may be reset during page navigation.
    When script with a matching URL is encountered, the cache is optionally
    produced upon backend discretion, based on internal heuristics.
    See also: ``Page.compilationCacheProduced``.

    **EXPERIMENTAL**

    :param scripts:
    """
    params: T_JSON_DICT = dict()
    params['scripts'] = [i.to_json() for i in scripts]
    cmd_dict: T_JSON_DICT = {'method': 'Page.produceCompilationCache', 'params': params}
    json = (yield cmd_dict)