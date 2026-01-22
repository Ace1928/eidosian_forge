import numpy as np
import shutil
from ..util.dtype import img_as_bool
from ._registry import registry, registry_urls
from .. import __version__
import os.path as osp
import os
def hubble_deep_field():
    """Hubble eXtreme Deep Field.

    This photograph contains the Hubble Telescope's farthest ever view of
    the universe. It can be useful as an example for multi-scale
    detection.

    Notes
    -----
    This image was downloaded from
    `HubbleSite
    <http://hubblesite.org/newscenter/archive/releases/2012/37/image/a/>`__.

    The image was captured by NASA and `may be freely used in the public domain
    <http://www.nasa.gov/audience/formedia/features/MP_Photo_Guidelines.html>`_.

    Returns
    -------
    hubble_deep_field : (872, 1000, 3) uint8 ndarray
        Hubble deep field image.
    """
    return _load('data/hubble_deep_field.jpg')