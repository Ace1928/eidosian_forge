from __future__ import annotations
import json
import pickle  # use pickle, not cPickle so that we get the traceback in case of errors
import string
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
from unittest import TestCase
import pytest
from monty.json import MontyDecoder, MontyEncoder, MSONable
from monty.serialization import loadfn
from pymatgen.core import ROOT, SETTINGS, Structure
def serialize_with_pickle(self, objects: Any, protocols: Sequence[int] | None=None, test_eq: bool=True):
    """Test whether the object(s) can be serialized and deserialized with
        pickle. This method tries to serialize the objects with pickle and the
        protocols specified in input. Then it deserializes the pickle format
        and compares the two objects with the __eq__ operator if
        test_eq is True.

        Args:
            objects: Object or list of objects.
            protocols: List of pickle protocols to test. If protocols is None,
                HIGHEST_PROTOCOL is tested.
            test_eq: If True, the deserialized object is compared with the
                original object using the __eq__ method.

        Returns:
            Nested list with the objects deserialized with the specified
            protocols.
        """
    got_single_object = False
    if not isinstance(objects, (list, tuple)):
        got_single_object = True
        objects = [objects]
    protocols = protocols or [pickle.HIGHEST_PROTOCOL]
    objects_by_protocol, errors = ([], [])
    for protocol in protocols:
        tmpfile = self.tmp_path / f'tempfile_{protocol}.pkl'
        try:
            with open(tmpfile, 'wb') as file:
                pickle.dump(objects, file, protocol=protocol)
        except Exception as exc:
            errors.append(f'pickle.dump with protocol={protocol!r} raised:\n{exc}')
            continue
        try:
            with open(tmpfile, 'rb') as file:
                unpickled_objs = pickle.load(file)
        except Exception as exc:
            errors.append(f'pickle.load with protocol={protocol!r} raised:\n{exc}')
            continue
        if test_eq:
            for orig, unpickled in zip(objects, unpickled_objs):
                assert orig == unpickled, f'Unpickled and original objects are unequal for protocol={protocol!r}\norig={orig!r}\nunpickled={unpickled!r}'
        objects_by_protocol.append(unpickled_objs)
    if errors:
        raise ValueError('\n'.join(errors))
    if got_single_object:
        return [o[0] for o in objects_by_protocol]
    return objects_by_protocol