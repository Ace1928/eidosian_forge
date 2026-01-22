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
def tag_sents(self, sentences):
    """
        Tag multiple sentences.

        Takes multiple sentences as a list where each sentence is a list of
        tokens.

        :param sentences: Input sentences to tag
        :type sentences: list(list(str))
        :rtype: list(list(tuple(str, str))
        """
    sentences = (' '.join(words) for words in sentences)
    return [sentences[0] for sentences in self.raw_tag_sents(sentences)]