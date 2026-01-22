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
def find_malt_model(model_filename):
    """
    A module to find pre-trained MaltParser model.
    """
    if model_filename is None:
        return 'malt_temp.mco'
    elif os.path.exists(model_filename):
        return model_filename
    else:
        return find_file(model_filename, env_vars=('MALT_MODEL',), verbose=False)