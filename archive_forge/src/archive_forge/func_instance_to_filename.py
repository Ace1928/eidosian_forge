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
def instance_to_filename(klass, img: FileBasedImage, filename: FileSpec) -> None:
    """Save `img` in our own format, to name implied by `filename`

        This is a class method

        Parameters
        ----------
        img : ``any FileBasedImage`` instance

        filename : str
           Filename, implying name to which to save image.
        """
    img = klass.from_image(img)
    img.to_filename(filename)