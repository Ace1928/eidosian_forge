import os
import re
import sys
import inspect
import logging
from abc import ABC, ABCMeta
from fnmatch import fnmatch
from pathlib import Path
from typing import Optional, List
from jinja2 import Environment, ChoiceLoader, FileSystemLoader, \
from elementpath import datatypes
import xmlschema
from xmlschema.validators import XsdType, XsdElement, XsdAttribute
from xmlschema.names import XSD_NAMESPACE
@staticmethod
@test_method
def multi_sequence(xsd_type):
    try:
        return any((e.is_multiple() for e in xsd_type.content.iter_elements()))
    except AttributeError:
        return False