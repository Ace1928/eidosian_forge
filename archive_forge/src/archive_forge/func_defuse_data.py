import sys
import os
import argparse
import logging
import pathlib
from urllib.error import URLError
import xmlschema
from xmlschema import XMLSchema, XMLSchema11, iter_errors, to_json, from_json, etree_tostring
from xmlschema.exceptions import XMLSchemaValueError
def defuse_data(value):
    if value not in ('always', 'remote', 'never'):
        raise argparse.ArgumentTypeError('%r is not a valid value' % value)
    return value