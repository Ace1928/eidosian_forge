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
def match_rule(fn, index, rule):
    if rule.ctx.filename != fn:
        return
    for prop, prp in rule.properties.items():
        if prp.line != index:
            continue
        yield prp
    for child in rule.children:
        for r in match_rule(fn, index, child):
            yield r
    if rule.canvas_root:
        for r in match_rule(fn, index, rule.canvas_root):
            yield r
    if rule.canvas_before:
        for r in match_rule(fn, index, rule.canvas_before):
            yield r
    if rule.canvas_after:
        for r in match_rule(fn, index, rule.canvas_after):
            yield r