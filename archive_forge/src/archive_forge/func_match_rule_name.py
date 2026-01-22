import sys
from os import environ
from os.path import join
from copy import copy
from types import CodeType
from functools import partial
from kivy.factory import Factory
from kivy.lang.parser import (
from kivy.logger import Logger
from kivy.utils import QueryDict
from kivy.cache import Cache
from kivy import kivy_data_dir
from kivy.context import register_context
from kivy.resources import resource_find
from kivy._event import Observable, EventDispatcher
def match_rule_name(self, rule_name):
    """Return a list of :class:`ParserRule` objects matching the widget.
        """
    cache = self._match_name_cache
    rule_name = str(rule_name)
    k = rule_name.lower()
    if k in cache:
        return cache[k]
    rules = []
    for selector, rule in self.rules:
        if selector.match_rule_name(rule_name):
            if rule.avoid_previous_rules:
                del rules[:]
            rules.append(rule)
    cache[k] = rules
    return rules