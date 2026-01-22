from __future__ import annotations
from dask.array import core
def from_tiledb(uri, attribute=None, chunks=None, storage_options=None, **kwargs):
    """Load array from the TileDB storage format

    See https://docs.tiledb.io for more information about TileDB.

    Parameters
    ----------
    uri: TileDB array or str
        Location to save the data
    attribute: str or None
        Attribute selection (single-attribute view on multi-attribute array)


    Returns
    -------

    A Dask Array

    Examples
    --------

    >>> import tempfile, tiledb
    >>> import dask.array as da, numpy as np
    >>> uri = tempfile.NamedTemporaryFile().name
    >>> _ = tiledb.from_numpy(uri, np.arange(0,9).reshape(3,3))  # create a tiledb array
    >>> tdb_ar = da.from_tiledb(uri)  # read back the array
    >>> tdb_ar.shape
    (3, 3)
    >>> tdb_ar.mean().compute()
    4.0
    """
    import tiledb
    tiledb_config = storage_options or dict()
    key = tiledb_config.pop('key', None)
    if isinstance(uri, tiledb.Array):
        tdb = uri
    else:
        tdb = tiledb.open(uri, attr=attribute, config=tiledb_config, key=key)
    if tdb.schema.sparse:
        raise ValueError('Sparse TileDB arrays are not supported')
    if not attribute:
        if tdb.schema.nattr > 1:
            raise TypeError("keyword 'attribute' must be providedwhen loading a multi-attribute TileDB array")
        else:
            attribute = tdb.schema.attr(0).name
    if tdb.iswritable:
        raise ValueError('TileDB array must be open for reading')
    chunks = chunks or _tiledb_to_chunks(tdb)
    assert len(chunks) == tdb.schema.ndim
    return core.from_array(tdb, chunks, name='tiledb-%s' % uri)