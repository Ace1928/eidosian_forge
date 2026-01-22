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
def unified(self):
    """
        Returns a new document containing all records having same identifiers
        unified (including those inside bundles).

        :return: :py:class:`ProvDocument`
        """
    document = ProvDocument(self._unified_records())
    document._namespaces = self._namespaces
    for bundle in self.bundles:
        unified_bundle = bundle.unified()
        document.add_bundle(unified_bundle)
    return document