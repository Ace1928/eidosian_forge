import base64
import json
import logging
from itertools import tee
import jmespath
from botocore.exceptions import PaginationError
from botocore.utils import merge_dicts, set_value_from_jmespath
def result_key_iters(self):
    teed_results = tee(self, len(self.result_keys))
    return [ResultKeyIterator(i, result_key) for i, result_key in zip(teed_results, self.result_keys)]