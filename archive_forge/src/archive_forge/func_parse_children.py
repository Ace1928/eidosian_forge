import json
import random
import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union
from typing import List as LList
from pydantic import BaseModel, ConfigDict, Field, validator
from pydantic.alias_generators import to_camel
@validator('children', pre=True, each_item=True)
def parse_children(cls, v):
    if isinstance(v, BaseModel):
        v = v.model_dump()
    if isinstance(v, dict):
        if v.get('type') == 'latex':
            return InlineLatex(**v)
        elif v.get('type') == 'link':
            return InlineLink(**v)
    return Text(**v)