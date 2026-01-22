import os
import re
import subprocess
import sys
import tempfile
from nltk.data import ZipFilePathPointer
from nltk.internals import find_dir
from nltk.tokenize.api import TokenizerI

        A module to find REPP tokenizer binary and its *repp.set* config file.
        