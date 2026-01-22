import re
from abc import ABC, abstractmethod
from typing import (
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import (
from langchain_core.retrievers import BaseRetriever
from typing_extensions import Annotated
def get_attribute_value(self) -> str:
    if not self.AdditionalAttributes:
        return ''
    if not self.AdditionalAttributes[0]:
        return ''
    else:
        return self.AdditionalAttributes[0].get_value_text()