from collections import defaultdict
from copy import deepcopy
import datetime
import io
import itertools
import logging
import os
import shutil
import tempfile
from urllib.parse import urlparse
import dateutil.parser
from prov import Error, serializers
from prov.constants import *
from prov.identifier import Identifier, QualifiedName, Namespace
def sorted_attributes(element, attributes):
    """
    Helper function sorting attributes into the order required by PROV-XML.

    :param element: The prov element used to derive the type and the
        attribute order for the type.
    :param attributes: The attributes to sort.
    """
    attributes = list(attributes)
    order = list(PROV_REC_CLS[element].FORMAL_ATTRIBUTES)
    order.extend([PROV_LABEL, PROV_LOCATION, PROV_ROLE, PROV_TYPE, PROV_VALUE])

    def sort_fct(x):
        return (str(x[0]), str(x[1].value if hasattr(x[1], 'value') else x[1]))
    sorted_elements = []
    for item in order:
        this_type_list = []
        for e in list(attributes):
            if e[0] != item:
                continue
            this_type_list.append(e)
            attributes.remove(e)
        this_type_list.sort(key=sort_fct)
        sorted_elements.extend(this_type_list)
    attributes.sort(key=sort_fct)
    sorted_elements.extend(attributes)
    return sorted_elements