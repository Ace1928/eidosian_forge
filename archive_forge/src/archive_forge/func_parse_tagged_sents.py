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
def parse_tagged_sents(self, sentences, verbose=False, top_relation_label='null'):
    """
        Use MaltParser to parse multiple POS tagged sentences. Takes multiple
        sentences where each sentence is a list of (word, tag) tuples.
        The sentences must have already been tokenized and tagged.

        :param sentences: Input sentences to parse
        :type sentence: list(list(tuple(str, str)))
        :return: iter(iter(``DependencyGraph``)) the dependency graph
            representation of each sentence
        """
    if not self._trained:
        raise Exception('Parser has not been trained. Call train() first.')
    with tempfile.NamedTemporaryFile(prefix='malt_input.conll.', dir=self.working_dir, mode='w', delete=False) as input_file:
        with tempfile.NamedTemporaryFile(prefix='malt_output.conll.', dir=self.working_dir, mode='w', delete=False) as output_file:
            for line in taggedsents_to_conll(sentences):
                input_file.write(str(line))
            input_file.close()
            cmd = self.generate_malt_command(input_file.name, output_file.name, mode='parse')
            _current_path = os.getcwd()
            try:
                os.chdir(os.path.split(self.model)[0])
            except:
                pass
            ret = self._execute(cmd, verbose)
            os.chdir(_current_path)
            if ret != 0:
                raise Exception('MaltParser parsing (%s) failed with exit code %d' % (' '.join(cmd), ret))
            with open(output_file.name) as infile:
                for tree_str in infile.read().split('\n\n'):
                    yield iter([DependencyGraph(tree_str, top_relation_label=top_relation_label)])
    os.remove(input_file.name)
    os.remove(output_file.name)