import sys
import os.path
from lxml import etree as _etree # due to validator __init__ signature
def schematron_schema_valid(arg):
    raise NotImplementedError('Validating the ISO schematron requires iso-schematron.rng')