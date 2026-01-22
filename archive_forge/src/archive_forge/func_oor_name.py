import functools
import gettext
import logging
import os
import shutil
import sys
import warnings
import xml.dom.minidom
import xml.parsers.expat
import zipfile
def oor_name(name, element):
    return element.attributes['oor:name'].value.lower() == name