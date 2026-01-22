import json
import struct
import zlib
import warnings
from io import BytesIO
import numpy as np
import symengine as sym
from symengine.lib.symengine_wrapper import (  # pylint: disable = no-name-in-module
from qiskit.exceptions import QiskitError
from qiskit.pulse import library, channels, instructions
from qiskit.pulse.schedule import ScheduleBlock
from qiskit.qpy import formats, common, type_keys
from qiskit.qpy.binary_io import value
from qiskit.qpy.exceptions import QpyError
from qiskit.pulse.configuration import Kernel, Discriminator
def read_schedule_block(file_obj, version, metadata_deserializer=None, use_symengine=False):
    """Read a single ScheduleBlock from the file like object.

    Args:
        file_obj (File): A file like object that contains the QPY binary data.
        version (int): QPY version.
        metadata_deserializer (JSONDecoder): An optional JSONDecoder class
            that will be used for the ``cls`` kwarg on the internal
            ``json.load`` call used to deserialize the JSON payload used for
            the :attr:`.ScheduleBlock.metadata` attribute for a schdule block
            in the file-like object. If this is not specified the circuit metadata will
            be parsed as JSON with the stdlib ``json.load()`` function using
            the default ``JSONDecoder`` class.
        use_symengine (bool): If True, symbolic objects will be serialized using symengine's
            native mechanism. This is a faster serialization alternative, but not supported in all
            platforms. Please check that your target platform is supported by the symengine library
            before setting this option, as it will be required by qpy to deserialize the payload.
    Returns:
        ScheduleBlock: The schedule block object from the file.

    Raises:
        TypeError: If any of the instructions is invalid data format.
        QiskitError: QPY version is earlier than block support.
    """
    if version < 5:
        raise QiskitError(f'QPY version {version} does not support ScheduleBlock.')
    data = formats.SCHEDULE_BLOCK_HEADER._make(struct.unpack(formats.SCHEDULE_BLOCK_HEADER_PACK, file_obj.read(formats.SCHEDULE_BLOCK_HEADER_SIZE)))
    name = file_obj.read(data.name_size).decode(common.ENCODE)
    metadata_raw = file_obj.read(data.metadata_size)
    metadata = json.loads(metadata_raw, cls=metadata_deserializer)
    context = _read_alignment_context(file_obj, version)
    block = ScheduleBlock(name=name, metadata=metadata, alignment_context=context)
    for _ in range(data.num_elements):
        block_elm = _read_element(file_obj, version, metadata_deserializer, use_symengine)
        block.append(block_elm, inplace=True)
    if version >= 7:
        flat_key_refdict = common.read_mapping(file_obj=file_obj, deserializer=_loads_reference_item, version=version, metadata_deserializer=metadata_deserializer)
        ref_dict = {}
        for key_str, schedule in flat_key_refdict.items():
            if schedule is not None:
                composite_key = tuple(key_str.split(instructions.Reference.key_delimiter))
                ref_dict[composite_key] = schedule
        if ref_dict:
            block.assign_references(ref_dict, inplace=True)
    return block