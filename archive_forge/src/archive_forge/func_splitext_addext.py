from __future__ import annotations
import os
import pathlib
import typing as ty
def splitext_addext(filename: FileSpec, addexts: ty.Sequence[str]=('.gz', '.bz2', '.zst'), match_case: bool=False) -> tuple[str, str, str]:
    """Split ``/pth/fname.ext.gz`` into ``/pth/fname, .ext, .gz``

    where ``.gz`` may be any of passed `addext` trailing suffixes.

    Parameters
    ----------
    filename : str or os.PathLike
       filename that may end in any or none of `addexts`
    match_case : bool, optional
       If True, match case of `addexts` and `filename`, otherwise do
       case-insensitive match.

    Returns
    -------
    froot : str
       Root of filename - e.g. ``/pth/fname`` in example above
    ext : str
       Extension, where extension is not in `addexts` - e.g. ``.ext`` in
       example above
    addext : str
       Any suffixes appearing in `addext` occurring at end of filename

    Examples
    --------
    >>> splitext_addext('fname.ext.gz')
    ('fname', '.ext', '.gz')
    >>> splitext_addext('fname.ext')
    ('fname', '.ext', '')
    >>> splitext_addext('fname.ext.foo', ('.foo', '.bar'))
    ('fname', '.ext', '.foo')
    """
    filename = _stringify_path(filename)
    if match_case:
        endswith = _endswith
    else:
        endswith = _iendswith
    for ext in addexts:
        if endswith(filename, ext):
            extpos = -len(ext)
            filename, addext = (filename[:extpos], filename[extpos:])
            break
    else:
        addext = ''
    extpos = filename.rfind('.')
    if extpos < 0 or filename.strip('.') == '':
        root, ext = (filename, '')
    else:
        root, ext = (filename[:extpos], filename[extpos:])
    return (root, ext, addext)