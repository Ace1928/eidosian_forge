from __future__ import absolute_import, unicode_literals
import re
from abc import ABCMeta, abstractmethod
from unittest import TestCase
import pytest
import six
from pybtex import textutils
from pybtex.richtext import HRef, Protected, String, Symbol, Tag, Text, nbsp
def test_render_as(self):
    assert nbsp.render_as('latex') == '~'
    assert nbsp.render_as('html') == '&nbsp;'
    assert Text(nbsp, nbsp).render_as('html') == '&nbsp;&nbsp;'