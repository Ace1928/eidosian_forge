import time
from collections import defaultdict
from unittest import TextTestResult as _TextTestResult
from unittest import TextTestRunner
from scrapy.commands import ScrapyCommand
from scrapy.contracts import ContractsManager
from scrapy.utils.conf import build_component_list
from scrapy.utils.misc import load_object, set_environ
def short_desc(self):
    return 'Check spider contracts'