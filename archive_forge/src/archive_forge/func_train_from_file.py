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
def train_from_file(self, conll_file, verbose=False):
    """
        Train MaltParser from a file
        :param conll_file: str for the filename of the training input data
        :type conll_file: str
        """
    if isinstance(conll_file, ZipFilePathPointer):
        with tempfile.NamedTemporaryFile(prefix='malt_train.conll.', dir=self.working_dir, mode='w', delete=False) as input_file:
            with conll_file.open() as conll_input_file:
                conll_str = conll_input_file.read()
                input_file.write(str(conll_str))
            return self.train_from_file(input_file.name, verbose=verbose)
    cmd = self.generate_malt_command(conll_file, mode='learn')
    ret = self._execute(cmd, verbose)
    if ret != 0:
        raise Exception('MaltParser training (%s) failed with exit code %d' % (' '.join(cmd), ret))
    self._trained = True