import numpy as np
from .core.imopen import imopen
def immeta(uri, *, index=None, plugin=None, extension=None, exclude_applied=True, **kwargs):
    """Read format-specific metadata.

    Opens the given URI and reads metadata for an ndimage from it. The contents
    of the returned metadata dictionary is specific to both the image format and
    plugin used to open the ImageResource. To learn about the exact behavior,
    check the documentation of the relevant plugin. Typically, immeta returns a
    dictionary specific to the image format, where keys match metadata field
    names and values are a field's contents.

    Parameters
    ----------
    uri : {str, pathlib.Path, bytes, file}
        The resource to load the image from, e.g. a filename, pathlib.Path, http
        address or file object, see the docs for more info.
    index : {int, None}
        If the ImageResource contains multiple ndimages, and index is an
        integer, select the index-th ndimage from among them and return its
        metadata. If index is an ellipsis (...), return global metadata. If
        index is None, let the plugin decide the default.
    plugin : {str, None}
        The plugin to be used. If None (default), performs a search for a
        matching plugin.
    extension : str
        If not None, treat the provided ImageResource as if it had the given
        extension. This affects the order in which backends are considered.
    **kwargs :
        Additional keyword arguments will be passed to the plugin's metadata
        method.

    Returns
    -------
    image : ndimage
        The ndimage located at the given URI.

    """
    plugin_kwargs = {'legacy_mode': False, 'plugin': plugin, 'extension': extension}
    call_kwargs = kwargs
    call_kwargs['exclude_applied'] = exclude_applied
    if index is not None:
        call_kwargs['index'] = index
    with imopen(uri, 'r', **plugin_kwargs) as img_file:
        metadata = img_file.metadata(**call_kwargs)
    return metadata