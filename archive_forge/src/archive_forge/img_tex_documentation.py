import json
from struct import unpack
from kivy.logger import Logger
from kivy.core.image import ImageLoaderBase, ImageData, ImageLoader

        if len(dds.images) > 1:
            images = dds.images
            images_size = dds.images_size
            for index in range(1, len(dds.images)):
                w, h = images_size[index]
                data = images[index]
                im.add_mipmap(index, w, h, data)
        