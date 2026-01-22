import os
import warnings
import pkgutil
from plotly.optional_imports import get_module
from plotly import tools
from ._plotlyjs_version import __plotlyjs_version__
def build_save_image_post_script(image, image_filename, image_height, image_width, caller):
    if image:
        if image not in __IMAGE_FORMATS:
            raise ValueError('The image parameter must be one of the following: {}'.format(__IMAGE_FORMATS))
        script = get_image_download_script(caller)
        post_script = script.format(format=image, width=image_width, height=image_height, filename=image_filename)
    else:
        post_script = None
    return post_script