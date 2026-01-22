import collections
from enum import Enum
import json
import os
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Union
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
def tag_searcher(searcher: Union['BasicVariantGenerator', 'Searcher']):
    from ray.tune.search import BasicVariantGenerator, Searcher
    if isinstance(searcher, BasicVariantGenerator):
        record_extra_usage_tag(TagKey.TUNE_SEARCHER, 'BasicVariantGenerator')
    elif isinstance(searcher, Searcher):
        searcher_name = _find_class_name(searcher, 'ray.tune.search', TUNE_SEARCHERS.union(TUNE_SEARCHER_WRAPPERS))
        if searcher_name in TUNE_SEARCHER_WRAPPERS:
            return
        record_extra_usage_tag(TagKey.TUNE_SEARCHER, searcher_name)
    else:
        assert False, 'Not expecting a non-BasicVariantGenerator, non-Searcher type passed in for `tag_searcher`.'