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
def end_date(self):
    self.add_object(_date_from_string(self.get_data()))