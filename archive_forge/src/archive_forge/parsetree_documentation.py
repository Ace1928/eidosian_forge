import re
from mako import ast
from mako import exceptions
from mako import filters
from mako import util
construct a new Tag instance.

        this constructor not called directly, and is only called
        by subclasses.

        :param keyword: the tag keyword

        :param attributes: raw dictionary of attribute key/value pairs

        :param expressions: a set of identifiers that are legal attributes,
         which can also contain embedded expressions

        :param nonexpressions: a set of identifiers that are legal
         attributes, which cannot contain embedded expressions

        :param \**kwargs:
         other arguments passed to the Node superclass (lineno, pos)

        