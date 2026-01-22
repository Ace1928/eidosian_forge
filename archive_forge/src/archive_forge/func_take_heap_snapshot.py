from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
def take_heap_snapshot(report_progress: typing.Optional[bool]=None, treat_global_objects_as_roots: typing.Optional[bool]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    :param report_progress: *(Optional)* If true 'reportHeapSnapshotProgress' events will be generated while snapshot is being taken.
    :param treat_global_objects_as_roots: *(Optional)* If true, a raw snapshot without artifical roots will be generated
    """
    params: T_JSON_DICT = dict()
    if report_progress is not None:
        params['reportProgress'] = report_progress
    if treat_global_objects_as_roots is not None:
        params['treatGlobalObjectsAsRoots'] = treat_global_objects_as_roots
    cmd_dict: T_JSON_DICT = {'method': 'HeapProfiler.takeHeapSnapshot', 'params': params}
    json = (yield cmd_dict)