from typing import Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.tools import BaseTool
from langchain_community.utilities.polygon import PolygonAPIWrapper
Use the Polygon API tool.