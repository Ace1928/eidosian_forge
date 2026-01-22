from __future__ import annotations
import io
import typing as ty
from copy import deepcopy
from urllib import request
from ._compression import COMPRESSION_ERRORS
from .fileholders import FileHolder, FileMap
from .filename_parser import TypesFilenamesError, _stringify_path, splitext_addext, types_filenames
from .openers import ImageOpener
@classmethod
def path_maybe_image(klass, filename: FileSpec, sniff: FileSniff | None=None, sniff_max: int=1024) -> tuple[bool, FileSniff | None]:
    """Return True if `filename` may be image matching this class

        Parameters
        ----------
        filename : str or os.PathLike
            Filename for an image, or an image header (metadata) file.
            If `filename` points to an image data file, and the image type has
            a separate "header" file, we work out the name of the header file,
            and read from that instead of `filename`.
        sniff : None or (bytes, filename), optional
            Bytes content read from a previous call to this method, on another
            class, with metadata filename.  This allows us to read metadata
            bytes once from the image or header, and pass this read set of
            bytes to other image classes, therefore saving a repeat read of the
            metadata.  `filename` is used to validate that metadata would be
            read from the same file, re-reading if not.  None forces this
            method to read the metadata.
        sniff_max : int, optional
            The maximum number of bytes to read from the metadata.  If the
            metadata file is long enough, we read this many bytes from the
            file, otherwise we read to the end of the file.  Longer values
            sniff more of the metadata / image file, making it more likely that
            the returned sniff will be useful for later calls to
            ``path_maybe_image`` for other image classes.

        Returns
        -------
        maybe_image : bool
            True if `filename` may be valid for an image of this class.
        sniff : None or (bytes, filename)
            Read bytes content from found metadata.  May be None if the file
            does not appear to have useful metadata.
        """
    froot, ext, trailing = splitext_addext(filename, klass._compressed_suffixes)
    if ext.lower() not in klass.valid_exts:
        return (False, sniff)
    if not hasattr(klass.header_class, 'may_contain_header'):
        return (True, sniff)
    if sniff is not None and len(sniff[0]) < klass._meta_sniff_len:
        sniff = None
    sniff = klass._sniff_meta_for(filename, max(klass._meta_sniff_len, sniff_max), sniff)
    if sniff is None or len(sniff[0]) < klass._meta_sniff_len:
        return (False, sniff)
    return (klass.header_class.may_contain_header(sniff[0]), sniff)