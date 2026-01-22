from __future__ import division, print_function
import sys
import os
import io
import re
import glob
import math
import zlib
import time
import json
import enum
import struct
import pathlib
import warnings
import binascii
import tempfile
import datetime
import threading
import collections
import multiprocessing
import concurrent.futures
import numpy
def pformat_xml(xml):
    """Return pretty formatted XML."""
    try:
        import lxml.etree as etree
        if not isinstance(xml, bytes):
            xml = xml.encode('utf-8')
        xml = etree.parse(io.BytesIO(xml))
        xml = etree.tostring(xml, pretty_print=True, xml_declaration=True, encoding=xml.docinfo.encoding)
        xml = bytes2str(xml)
    except Exception:
        if isinstance(xml, bytes):
            xml = bytes2str(xml)
        xml = xml.replace('><', '>\n<')
    return xml.replace('  ', ' ').replace('\t', ' ')