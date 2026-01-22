import json
import random
import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union
from typing import List as LList
from pydantic import BaseModel, ConfigDict, Field, validator
from pydantic.alias_generators import to_camel
@validator('spec', pre=True)
def parse_json(cls, v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            raise ValueError('invalid json')
    return v