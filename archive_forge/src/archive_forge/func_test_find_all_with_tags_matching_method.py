from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_find_all_with_tags_matching_method(self):

    def id_matches_name(tag):
        return tag.name == tag.get('id')
    tree = self.soup('<a id="a">Match 1.</a>\n                            <a id="1">Does not match.</a>\n                            <b id="b">Match 2.</a>')
    self.assert_selects(tree.find_all(id_matches_name), ['Match 1.', 'Match 2.'])