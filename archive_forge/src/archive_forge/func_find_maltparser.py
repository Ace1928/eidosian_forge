import inspect
import os
import subprocess
import sys
import tempfile
from nltk.data import ZipFilePathPointer
from nltk.internals import find_dir, find_file, find_jars_within_path
from nltk.parse.api import ParserI
from nltk.parse.dependencygraph import DependencyGraph
from nltk.parse.util import taggedsents_to_conll
def find_maltparser(parser_dirname):
    """
    A module to find MaltParser .jar file and its dependencies.
    """
    if os.path.exists(parser_dirname):
        _malt_dir = parser_dirname
    else:
        _malt_dir = find_dir(parser_dirname, env_vars=('MALT_PARSER',))
    malt_dependencies = ['', '', '']
    _malt_jars = set(find_jars_within_path(_malt_dir))
    _jars = {os.path.split(jar)[1] for jar in _malt_jars}
    malt_dependencies = {'log4j.jar', 'libsvm.jar', 'liblinear-1.8.jar'}
    assert malt_dependencies.issubset(_jars)
    assert any(filter(lambda i: i.startswith('maltparser-') and i.endswith('.jar'), _jars))
    return list(_malt_jars)