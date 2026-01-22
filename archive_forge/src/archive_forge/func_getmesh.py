from __future__ import annotations
import functools
import operator
import re
from typing import Protocol, Sequence, cast
from . import ExifTags, Image, ImagePalette
def getmesh(self, image: Image.Image) -> list[tuple[tuple[int, int, int, int], tuple[int, int, int, int, int, int, int, int]]]:
    ...