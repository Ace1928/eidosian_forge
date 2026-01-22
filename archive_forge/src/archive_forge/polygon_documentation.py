import json
from typing import Any, Dict, Optional
import requests
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.utils import get_from_dict_or_env

        Get aggregate bars for a stock over a given date range
        in custom time window sizes.

        /v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}
        