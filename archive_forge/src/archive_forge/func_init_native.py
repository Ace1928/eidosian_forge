import os
import re
import sys
from typing import Any, Dict, List
from sphinx.errors import ExtensionError, SphinxError
from sphinx.search import SearchLanguage
from sphinx.util import import_object
def init_native(self, options: Dict) -> None:
    param = '-Owakati'
    dict = options.get('dict')
    if dict:
        param += ' -d %s' % dict
    self.native = MeCab.Tagger(param)