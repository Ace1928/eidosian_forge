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
@property
def default_ns_uri(self):
    """
        Returns the default namespace's URI, if any.

        :return: URI as string.
        """
    default_ns = self._namespaces.get_default_namespace()
    return default_ns.uri if default_ns else None