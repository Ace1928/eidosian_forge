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
def begin_array(self, attrs):
    a = []
    self.add_object(a)
    self.stack.append(a)