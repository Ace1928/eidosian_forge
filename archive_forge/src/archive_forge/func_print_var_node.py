import sys
from _pydevd_bundle import pydevd_xml
from os.path import basename
from _pydev_bundle import pydev_log
from urllib.parse import unquote_plus
from _pydevd_bundle.pydevd_constants import IS_PY311_OR_GREATER
def print_var_node(xml_node, stream):
    name = xml_node.getAttribute('name')
    value = xml_node.getAttribute('value')
    val_type = xml_node.getAttribute('type')
    found_as = xml_node.getAttribute('found_as')
    stream.write('Name: ')
    stream.write(unquote_plus(name))
    stream.write(', Value: ')
    stream.write(unquote_plus(value))
    stream.write(', Type: ')
    stream.write(unquote_plus(val_type))
    if found_as:
        stream.write(', Found as: %s' % (unquote_plus(found_as),))
    stream.write('\n')