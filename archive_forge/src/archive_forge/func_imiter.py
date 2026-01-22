import numpy as np
from .core.imopen import imopen
def imiter(uri, *, plugin=None, extension=None, format_hint=None, **kwargs):
    """Read a sequence of ndimages from a URI.

    Returns an iterable that yields ndimages from the given URI. The exact
    behavior depends on both, the file type and plugin used to open the file.
    To learn about the exact behavior, check the documentation of the relevant
    plugin.

    Parameters
    ----------
    uri : {str, pathlib.Path, bytes, file}
        The resource to load the image from, e.g. a filename, pathlib.Path,
        http address or file object, see the docs for more info.
    plugin : {str, None}
        The plugin to use. If set to None (default) imiter will perform a
        search for a matching plugin. If not None, this takes priority over
        the provided format hint (if present).
    extension : str
        If not None, treat the provided ImageResource as if it had the given
        extension. This affects the order in which backends are considered.
    format_hint : str
        Deprecated. Use `extension` instead.
    **kwargs :
        Additional keyword arguments will be passed to the plugin's ``iter``
        call.

    Yields
    ------
    image : ndimage
        The next ndimage located at the given URI.

    """
    with imopen(uri, 'r', legacy_mode=False, plugin=plugin, format_hint=format_hint, extension=extension) as img_file:
        for image in img_file.iter(**kwargs):
            yield np.asarray(image)