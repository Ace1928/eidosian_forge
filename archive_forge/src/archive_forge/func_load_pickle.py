import builtins
import importlib
import importlib.machinery
import inspect
import io
import linecache
import os.path
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, Callable, cast, Dict, Iterable, List, Optional, Union
from weakref import WeakValueDictionary
import torch
from torch.serialization import _get_restore_location, _maybe_decode_ascii
from ._directory_reader import DirectoryReader
from ._importlib import (
from ._mangling import demangle, PackageMangler
from ._package_unpickler import PackageUnpickler
from .file_structure_representation import _create_directory_from_file_list, Directory
from .glob_group import GlobPattern
from .importer import Importer
def load_pickle(self, package: str, resource: str, map_location=None) -> Any:
    """Unpickles the resource from the package, loading any modules that are needed to construct the objects
        using :meth:`import_module`.

        Args:
            package (str): The name of module package (e.g. ``"my_package.my_subpackage"``).
            resource (str): The unique name for the resource.
            map_location: Passed to `torch.load` to determine how tensors are mapped to devices. Defaults to ``None``.

        Returns:
            Any: The unpickled object.
        """
    pickle_file = self._zipfile_path(package, resource)
    restore_location = _get_restore_location(map_location)
    loaded_storages = {}
    loaded_reduces = {}
    storage_context = torch._C.DeserializationStorageContext()

    def load_tensor(dtype, size, key, location, restore_location):
        name = f'{key}.storage'
        if storage_context.has_storage(name):
            storage = storage_context.get_storage(name, dtype)._typed_storage()
        else:
            tensor = self.zip_reader.get_storage_from_record('.data/' + name, size, dtype)
            if isinstance(self.zip_reader, torch._C.PyTorchFileReader):
                storage_context.add_storage(name, tensor)
            storage = tensor._typed_storage()
        loaded_storages[key] = restore_location(storage, location)

    def persistent_load(saved_id):
        assert isinstance(saved_id, tuple)
        typename = _maybe_decode_ascii(saved_id[0])
        data = saved_id[1:]
        if typename == 'storage':
            storage_type, key, location, size = data
            dtype = storage_type.dtype
            if key not in loaded_storages:
                load_tensor(dtype, size, key, _maybe_decode_ascii(location), restore_location)
            storage = loaded_storages[key]
            return torch.storage.TypedStorage(wrap_storage=storage._untyped_storage, dtype=dtype, _internal=True)
        elif typename == 'reduce_package':
            if len(data) == 2:
                func, args = data
                return func(self, *args)
            reduce_id, func, args = data
            if reduce_id not in loaded_reduces:
                loaded_reduces[reduce_id] = func(self, *args)
            return loaded_reduces[reduce_id]
        else:
            f"Unknown typename for persistent_load, expected 'storage' or 'reduce_package' but got '{typename}'"
    data_file = io.BytesIO(self.zip_reader.get_record(pickle_file))
    unpickler = self.Unpickler(data_file)
    unpickler.persistent_load = persistent_load

    @contextmanager
    def set_deserialization_context():
        self.storage_context = storage_context
        self.last_map_location = map_location
        try:
            yield
        finally:
            self.storage_context = None
            self.last_map_location = None
    with set_deserialization_context():
        result = unpickler.load()
    torch._utils._validate_loaded_sparse_tensors()
    return result