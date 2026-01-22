from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
def next_wasm_disassembly_chunk(stream_id: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, WasmDisassemblyChunk]:
    """
    Disassemble the next chunk of lines for the module corresponding to the
    stream. If disassembly is complete, this API will invalidate the streamId
    and return an empty chunk. Any subsequent calls for the now invalid stream
    will return errors.

    **EXPERIMENTAL**

    :param stream_id:
    :returns: The next chunk of disassembly.
    """
    params: T_JSON_DICT = dict()
    params['streamId'] = stream_id
    cmd_dict: T_JSON_DICT = {'method': 'Debugger.nextWasmDisassemblyChunk', 'params': params}
    json = (yield cmd_dict)
    return WasmDisassemblyChunk.from_json(json['chunk'])