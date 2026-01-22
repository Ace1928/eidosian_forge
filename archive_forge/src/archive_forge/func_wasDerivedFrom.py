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
def wasDerivedFrom(self, usedEntity, activity=None, generation=None, usage=None, attributes=None):
    """
        Creates a new derivation record for this entity from a used entity.

        :param usedEntity: Entity or a string identifier for the used entity.
        :param activity: Activity or string identifier of the activity involved in
            the derivation (default: None).
        :param generation: Optionally extra activity to state qualified derivation
            through an internal generation (default: None).
        :param usage: Optionally extra entity to state qualified derivation through
            an internal usage (default: None).
        :param attributes: Optional other attributes as a dictionary or list
            of tuples to be added to the record optionally (default: None).
        """
    self._bundle.derivation(self, usedEntity, activity, generation, usage, other_attributes=attributes)
    return self