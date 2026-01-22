import json
import os  # required for doctests
import re
import socket
import time
from typing import List, Tuple
from nltk.internals import _java_options, config_java, find_jar_iter, java
from nltk.parse.api import ParserI
from nltk.parse.dependencygraph import DependencyGraph
from nltk.tag.api import TaggerI
from nltk.tokenize.api import TokenizerI
from nltk.tree import Tree
def raw_parse_sents(self, sentences, verbose=False, properties=None, *args, **kwargs):
    """Parse multiple sentences.

        Takes multiple sentences as a list of strings. Each sentence will be
        automatically tokenized and tagged.

        :param sentences: Input sentences to parse.
        :type sentences: list(str)
        :rtype: iter(iter(Tree))

        """
    default_properties = {'ssplit.eolonly': 'true'}
    default_properties.update(properties or {})
    "\n        for sentence in sentences:\n            parsed_data = self.api_call(sentence, properties=default_properties)\n\n            assert len(parsed_data['sentences']) == 1\n\n            for parse in parsed_data['sentences']:\n                tree = self.make_tree(parse)\n                yield iter([tree])\n        "
    parsed_data = self.api_call('\n'.join(sentences), properties=default_properties)
    for parsed_sent in parsed_data['sentences']:
        tree = self.make_tree(parsed_sent)
        yield iter([tree])