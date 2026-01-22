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
def new_record(self, record_type, identifier, attributes=None, other_attributes=None):
    """
        Creates a new record.

        :param record_type: Type of record (one of :py:const:`PROV_REC_CLS`).
        :param identifier: Identifier for new record.
        :param attributes: Attributes as a dictionary or list of tuples to be added
            to the record optionally (default: None).
        :param other_attributes: Optional other attributes as a dictionary or list
            of tuples to be added to the record optionally (default: None).
        """
    attr_list = []
    if attributes:
        if isinstance(attributes, dict):
            attr_list.extend(((attr, value) for attr, value in attributes.items()))
        else:
            attr_list.extend(attributes)
    if other_attributes:
        attr_list.extend(other_attributes.items() if isinstance(other_attributes, dict) else other_attributes)
    new_record = PROV_REC_CLS[record_type](self, self.valid_qualified_name(identifier), attr_list)
    self._add_record(new_record)
    return new_record