import re
from typing import Any, Dict, List, Tuple, Union
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.pydantic_v1 import validator
from langchain.output_parsers.format_instructions import (
The Pandas DataFrame to parse.