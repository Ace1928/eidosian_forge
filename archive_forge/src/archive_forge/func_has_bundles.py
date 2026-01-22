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
def has_bundles(self):
    """
        `True` if the object has at least one bundle, `False` otherwise.

        :return: bool
        """
    return len(self._bundles) > 0