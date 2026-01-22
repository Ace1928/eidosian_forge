import re
from functools import total_ordering
from xml.etree import ElementTree
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.internals import raise_unorderable_types
from nltk.tree import Tree
def roleset(self, roleset_id):
    """
        :return: the xml description for the given roleset.
        """
    baseform = roleset_id.split('.')[0]
    framefile = 'frames/%s.xml' % baseform
    if framefile not in self._framefiles:
        raise ValueError('Frameset file for %s not found' % roleset_id)
    with self.abspath(framefile).open() as fp:
        etree = ElementTree.parse(fp).getroot()
    for roleset in etree.findall('predicate/roleset'):
        if roleset.attrib['id'] == roleset_id:
            return roleset
    raise ValueError(f'Roleset {roleset_id} not found in {framefile}')