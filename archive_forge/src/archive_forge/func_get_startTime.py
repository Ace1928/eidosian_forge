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
def get_startTime(self):
    """
        Returns the time the activity started.

        :return: :py:class:`datetime.datetime`
        """
    values = self._attributes[PROV_ATTR_STARTTIME]
    return first(values) if values else None