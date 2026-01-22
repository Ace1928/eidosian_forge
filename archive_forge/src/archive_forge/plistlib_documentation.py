import binascii
import codecs
import datetime
import enum
from io import BytesIO
import itertools
import os
import re
import struct
from xml.parsers.expat import ParserCreate

        read the object by reference.

        May recursively read sub-objects (content of an array/dict/set)
        