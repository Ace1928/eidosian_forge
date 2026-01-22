from kivy.logger import Logger
from kivy.core.image import ImageLoaderBase, ImageData, ImageLoader
@staticmethod
def can_save(fmt, is_bytesio):
    if is_bytesio:
        return False
    return fmt in ImageLoaderPIL.extensions()