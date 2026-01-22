import datetime
from ..core import Format
from ..core.request import URI_BYTES, URI_FILE
import numpy as np
import warnings
Provides support for a wide range of Tiff images using the tifffile
    backend.

    Images that contain multiple pages can be read using ``imageio.mimread()``
    to read the individual pages, or ``imageio.volread()`` to obtain a
    single (higher dimensional) array.

    Note that global metadata is stored with the first frame in a TIFF file.
    Thus calling :py:meth:`Format.Writer.set_meta_data` after the first frame
    was written has no effect. Also, global metadata is ignored if metadata is
    provided via the `meta` argument of :py:meth:`Format.Writer.append_data`.

    If you have installed tifffile as a Python package, imageio will attempt
    to use that as backend instead of the bundled backend. Doing so can
    provide access to new performance improvements and bug fixes.

    Parameters for reading
    ----------------------
    offset : int
        Optional start position of embedded file. By default this is
        the current file position.
    size : int
        Optional size of embedded file. By default this is the number
        of bytes from the 'offset' to the end of the file.
    multifile : bool
        If True (default), series may include pages from multiple files.
        Currently applies to OME-TIFF only.
    multifile_close : bool
        If True (default), keep the handles of other files in multifile
        series closed. This is inefficient when few files refer to
        many pages. If False, the C runtime may run out of resources.

    Parameters for saving
    ---------------------
    bigtiff : bool
        If True, the BigTIFF format is used.
    byteorder : {'<', '>'}
        The endianness of the data in the file.
        By default this is the system's native byte order.
    software : str
        Name of the software used to create the image.
        Saved with the first page only.

    Metadata for reading
    --------------------
    planar_configuration : {'contig', 'planar'}
        Specifies if samples are stored contiguous or in separate planes.
        By default this setting is inferred from the data shape.
        'contig': last dimension contains samples.
        'planar': third last dimension contains samples.
    resolution_unit : (float, float) or ((int, int), (int, int))
        X and Y resolution in dots per inch as float or rational numbers.
    compression : int
        Value indicating the compression algorithm used, e.g. 5 is LZW,
        7 is JPEG, 8 is deflate.
        If 1, data are uncompressed.
    predictor : int
        Value 2 indicates horizontal differencing was used before compression,
        while 3 indicates floating point horizontal differencing.
        If 1, no prediction scheme was used before compression.
    orientation : {'top_left', 'bottom_right', ...}
        Oriented of image array.
    is_rgb : bool
        True if page contains a RGB image.
    is_contig : bool
        True if page contains a contiguous image.
    is_tiled : bool
        True if page contains tiled image.
    is_palette : bool
        True if page contains a palette-colored image and not OME or STK.
    is_reduced : bool
        True if page is a reduced image of another image.
    is_shaped : bool
        True if page contains shape in image_description tag.
    is_fluoview : bool
        True if page contains FluoView MM_STAMP tag.
    is_nih : bool
        True if page contains NIH image header.
    is_micromanager : bool
        True if page contains Micro-Manager metadata.
    is_ome : bool
        True if page contains OME-XML in image_description tag.
    is_sgi : bool
        True if page contains SGI image and tile depth tags.
    is_stk : bool
        True if page contains UIC2Tag tag.
    is_mdgel : bool
        True if page contains md_file_tag tag.
    is_mediacy : bool
        True if page contains Media Cybernetics Id tag.
    is_stk : bool
        True if page contains UIC2Tag tag.
    is_lsm : bool
        True if page contains LSM CZ_LSM_INFO tag.
    description : str
        Image description
    description1 : str
        Additional description
    is_imagej : None or str
        ImageJ metadata
    software : str
        Software used to create the TIFF file
    datetime : datetime.datetime
        Creation date and time

    Metadata for writing
    --------------------
    photometric : {'minisblack', 'miniswhite', 'rgb'}
        The color space of the image data.
        By default this setting is inferred from the data shape.
    planarconfig : {'contig', 'planar'}
        Specifies if samples are stored contiguous or in separate planes.
        By default this setting is inferred from the data shape.
        'contig': last dimension contains samples.
        'planar': third last dimension contains samples.
    resolution : (float, float) or ((int, int), (int, int))
        X and Y resolution in dots per inch as float or rational numbers.
    description : str
        The subject of the image. Saved with the first page only.
    compress : int
        Values from 0 to 9 controlling the level of zlib (deflate) compression.
        If 0, data are written uncompressed (default).
    predictor : bool
        If True, horizontal differencing is applied before compression.
        Note that using an int literal 1 actually means no prediction scheme
        will be used.
    volume : bool
        If True, volume data are stored in one tile (if applicable) using
        the SGI image_depth and tile_depth tags.
        Image width and depth must be multiple of 16.
        Few software can read this format, e.g. MeVisLab.
    writeshape : bool
        If True, write the data shape to the image_description tag
        if necessary and no other description is given.
    extratags: sequence of tuples
        Additional tags as [(code, dtype, count, value, writeonce)].

        code : int
            The TIFF tag Id.
        dtype : str
            Data type of items in 'value' in Python struct format.
            One of B, s, H, I, 2I, b, h, i, f, d, Q, or q.
        count : int
            Number of data values. Not used for string values.
        value : sequence
            'Count' values compatible with 'dtype'.
        writeonce : bool
            If True, the tag is written to the first page only.
    