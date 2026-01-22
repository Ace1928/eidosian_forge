import os
from lxml import etree
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def make_oneline(s):
    return etree.tostring(etree.XML(s)).replace(b'\n', b'')