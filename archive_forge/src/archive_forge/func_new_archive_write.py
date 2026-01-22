from contextlib import contextmanager
from ctypes import byref, cast, c_char, c_size_t, c_void_p, POINTER
from posixpath import join
import warnings
from . import ffi
from .entry import ArchiveEntry, FileType
from .ffi import (
@contextmanager
def new_archive_write(format_name, filter_name=None, options='', passphrase=None):
    archive_p = ffi.write_new()
    try:
        ffi.get_write_format_function(format_name)(archive_p)
        if filter_name:
            ffi.get_write_filter_function(filter_name)(archive_p)
        if passphrase and 'encryption' not in options:
            if format_name == 'zip':
                warnings.warn("The default encryption scheme of zip archives is weak. Use `options='encryption=$type'` to specify the encryption type you want to use. The supported values are 'zipcrypt' (the weak default), 'aes128' and 'aes256'.")
            options += ',encryption' if options else 'encryption'
        if options:
            if not isinstance(options, bytes):
                options = options.encode('utf-8')
            ffi.write_set_options(archive_p, options)
        if passphrase:
            if not isinstance(passphrase, bytes):
                passphrase = passphrase.encode('utf-8')
            try:
                ffi.write_set_passphrase(archive_p, passphrase)
            except AttributeError:
                raise NotImplementedError(f"the libarchive being used (version {ffi.version_number()}, path {ffi.libarchive_path}) doesn't support encryption")
        yield archive_p
        ffi.write_close(archive_p)
        ffi.write_free(archive_p)
    except Exception:
        ffi.write_fail(archive_p)
        ffi.write_free(archive_p)
        raise

    @property
    def bytes_written(self):
        return ffi.filter_bytes(self._pointer, -1)