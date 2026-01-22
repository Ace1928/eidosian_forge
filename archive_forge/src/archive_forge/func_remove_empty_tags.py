import difflib
import glob
import inspect
import io
from lxml import etree
import os
import unittest
import warnings
from prov.identifier import Namespace, QualifiedName
from prov.constants import PROV
import prov.model as prov
from prov.tests.test_model import AllTestsBase
from prov.tests.utility import RoundTripTestCase
def remove_empty_tags(tree):
    if tree.text is not None and tree.text.strip() == '':
        tree.text = None
    for elem in tree:
        if etree.iselement(elem):
            remove_empty_tags(elem)