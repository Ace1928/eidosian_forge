import sys
import os
import os.path
import tempfile
import zipfile
from xml.dom import minidom
import time
import re
import copy
import itertools
import docutils
from docutils import frontend, nodes, utils, writers, languages
from docutils.readers import standalone
from docutils.transforms import references
def get_image_scaled_width_height(self, node, source):
    """Return the image size in centimeters adjusted by image attrs."""
    scale = self.get_image_scale(node)
    width, width_unit = self.get_image_width_height(node, 'width')
    height, _ = self.get_image_width_height(node, 'height')
    dpi = (72, 72)
    if PIL is not None and source in self.image_dict:
        filename, destination = self.image_dict[source]
        imageobj = PIL.Image.open(filename, 'r')
        dpi = imageobj.info.get('dpi', dpi)
        try:
            iter(dpi)
        except TypeError:
            dpi = (dpi, dpi)
    else:
        imageobj = None
    if width is None or height is None:
        if imageobj is None:
            raise RuntimeError('image size not fully specified and PIL not installed')
        if width is None:
            width = imageobj.size[0]
            width = float(width) * 0.026
        if height is None:
            height = imageobj.size[1]
            height = float(height) * 0.026
        if width_unit == '%':
            factor = width
            image_width = imageobj.size[0]
            image_width = float(image_width) * 0.026
            image_height = imageobj.size[1]
            image_height = float(image_height) * 0.026
            line_width = self.get_page_width()
            width = factor * line_width
            factor = factor * line_width / image_width
            height = factor * image_height
    width *= scale
    height *= scale
    width = '%.2fcm' % width
    height = '%.2fcm' % height
    return (width, height)