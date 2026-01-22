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
def rewrite_query(interface, join_field, common_field, _filter):
    _new_filter = []
    for _f in _filter:
        if isinstance(_f, list):
            _new_filter.append(rewrite_query(interface, join_field, common_field, _f))
        elif isinstance(_f, tuple):
            _datatype = _f[0].split('/')[0]
            _res = interface.select(_datatype, ['%s/%s' % (_datatype, common_field)]).where([_f, 'AND'])
            _new_f = [(join_field, '=', '%s' % sid) for sid in _res['subject_id']]
            _new_f.append('OR')
            _new_filter.append(_new_f)
        elif isinstance(_f, str):
            _new_filter.append(_f)
        else:
            raise Exception('Invalid filter')
    return _new_filter