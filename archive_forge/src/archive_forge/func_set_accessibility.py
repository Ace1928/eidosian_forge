import lxml
import os
import os.path as op
import sys
import re
import shutil
import tempfile
import zipfile
import codecs
from fnmatch import fnmatch
from itertools import islice
from lxml import etree
from pathlib import Path
from .uriutil import join_uri, translate_uri, uri_segment
from .uriutil import uri_last, uri_nextlast
from .uriutil import uri_parent, uri_grandparent
from .uriutil import uri_shape
from .uriutil import file_path
from .jsonutil import JsonTable, get_selection
from .pathutil import find_files, ensure_dir_exists
from .attributes import EAttrs
from .search import rpn_contraints, query_from_xml
from .errors import is_xnat_error, parse_put_error_message
from .errors import DataError, ProgrammingError, catch_error
from .provenance import Provenance
from .pipelines import Pipelines
from . import schema
from . import httputil
from . import downloadutils
from . import derivatives
import types
import pkgutil
import inspect
from urllib.parse import quote, unquote
def set_accessibility(self, accessibility='protected'):
    """ Sets project accessibility.

            .. note::
                Write access is given or not by the user level for a
                specific project.

            Parameters
            ----------
            accessibility: public | protected | private
                Sets the project accessibility:
                    - public: the project is visible and provides read
                      access for anyone.
                    - protected: the project is visible by anyone but the
                      data is accessible for allowed users only.
                    - private: the project is visible by allowed users only.

        """
    return self._intf._exec(join_uri(self._uri, 'accessibility', accessibility), 'PUT')