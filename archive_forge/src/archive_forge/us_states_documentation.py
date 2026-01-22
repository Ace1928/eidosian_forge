from __future__ import annotations
import logging # isort:skip
import codecs
import csv
import gzip
import xml.etree.ElementTree as et
from math import nan
from typing import TYPE_CHECKING, TypedDict
import numpy as np
from ..util.sampledata import package_path


    