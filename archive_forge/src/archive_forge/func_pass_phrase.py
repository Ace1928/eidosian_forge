import abc
import functools
import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
@pass_phrase.setter
@immutable_after_save
def pass_phrase(self, value):
    self._meta['pass_phrase'] = value