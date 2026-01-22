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
def specializationOf(self, generalEntity):
    """
        Creates a new specialisation record for this from a general entity.

        :param generalEntity: Entity or a string identifier for the general entity.
        """
    self._bundle.specialization(self, generalEntity)
    return self