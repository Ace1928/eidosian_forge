import numpy as np
import shutil
from ..util.dtype import img_as_bool
from ._registry import registry, registry_urls
from .. import __version__
import os.path as osp
import os
def gravel():
    """Gravel

    Returns
    -------
    gravel : (512, 512) uint8 image
        Grayscale gravel sample.

    Notes
    -----
    The original image was downloaded from
    `CC0Textures <https://cc0textures.com/view.php?tex=Gravel04>`__ and
    licensed under the Creative Commons CC0 License.

    The downloaded image was then rescaled to ``(1024, 1024)``, then the
    top left ``(512, 512)`` pixel region  was cropped prior to converting the
    image to grayscale and uint8 data type. The result was saved using the
    PNG format.
    """
    "\n    The following code was used to obtain the final image.\n\n    >>> import sys; print(sys.version)\n    >>> import platform; print(platform.platform())\n    >>> import skimage; print(f'scikit-image version: {skimage.__version__}')\n    >>> import numpy; print(f'numpy version: {numpy.__version__}')\n    >>> import imageio; print(f'imageio version {imageio.__version__}')\n    3.7.3 | packaged by conda-forge | (default, Jul  1 2019, 21:52:21)\n    [GCC 7.3.0]\n    Linux-5.0.0-20-generic-x86_64-with-debian-buster-sid\n    scikit-image version: 0.16.dev0\n    numpy version: 1.16.4\n    imageio version 2.4.1\n\n    >>> import requests\n    >>> import zipfile\n\n    >>> url = 'https://cdn.struffelproductions.com/file/cc0textures/Gravel04/%5B2K%5DGravel04.zip'\n    >>> r = requests.get(url)\n    >>> with open('[2K]Gravel04.zip', 'bw') as f:\n    ...     f.write(r.content)\n\n    >>> with zipfile.ZipFile('[2K]Gravel04.zip') as z:\n    ...     z.extract('Gravel04_col.jpg')\n\n    >>> from skimage.transform import resize\n    >>> gravel_orig = imageio.imread('Gravel04_col.jpg')\n    >>> gravel = resize(gravel_orig, (1024, 1024))\n    >>> gravel = skimage.img_as_ubyte(skimage.color.rgb2gray(gravel[:512, :512]))\n    >>> imageio.imwrite('gravel.png', gravel)\n    "
    return _load('data/gravel.png', as_gray=True)