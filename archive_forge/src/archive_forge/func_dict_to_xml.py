from __future__ import absolute_import, division, print_function
import json
import sys
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import string_types
def dict_to_xml(data, full_document=False):
    """
    Converts dict object to a valid XML string
    :param data: Python dict object
    :param full_document: When set to True the will have exactly one root.
    :return: Valid XML string
    """
    if not HAS_XMLTODICT:
        msg = "dict to xml conversion requires 'xmltodict' for given data %s ." % data
        raise Exception(msg + missing_required_lib('xmltodict'))
    try:
        return xmltodict.unparse(data, full_document=full_document)
    except Exception as exc:
        error = "'xmltodict' returned the following error when converting %s to xml. " % data
        raise Exception(error + to_native(exc, errors='surrogate_then_replace'))