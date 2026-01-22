import doctest
import os
import shutil
import tempfile
import unittest
from genshi.core import TEXT
from genshi.template.loader import TemplateLoader
from genshi.template.markup import MarkupTemplate
def template_loaded(template):

    def my_filter(stream, ctxt):
        for kind, data, pos in stream:
            if kind is TEXT and data.strip():
                data = ', '.join([data, data.lower()])
            yield (kind, data, pos)
    template.filters.insert(0, my_filter)