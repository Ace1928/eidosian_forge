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
def provn_representation(self):
    if self._langtag:
        return '%s@%s' % (_ensure_multiline_string_triple_quoted(self._value), str(self._langtag))
    else:
        return '%s %%%% %s' % (_ensure_multiline_string_triple_quoted(self._value), str(self._datatype))