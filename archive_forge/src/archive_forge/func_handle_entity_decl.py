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
def handle_entity_decl(self, entity_name, is_parameter_entity, value, base, system_id, public_id, notation_name):
    raise InvalidFileException('XML entity declarations are not supported in plist files')