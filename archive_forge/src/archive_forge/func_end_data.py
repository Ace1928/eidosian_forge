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
def end_data(self):
    self.add_object(_decode_base64(self.get_data()))