from __future__ import annotations
import traceback
import warnings
from collections import defaultdict
from types import ModuleType
from typing import TYPE_CHECKING, DefaultDict, Dict, List, Tuple, Type
from zope.interface import implementer
from scrapy import Request, Spider
from scrapy.interfaces import ISpiderLoader
from scrapy.settings import BaseSettings
from scrapy.utils.misc import walk_modules
from scrapy.utils.spider import iter_spider_classes

        Return a list with the names of all spiders available in the project.
        