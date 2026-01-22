from __future__ import annotations
import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import (
from uuid import UUID, uuid4
from langchain_core.pydantic_v1 import BaseModel
def draw_ascii(self) -> str:
    from langchain_core.runnables.graph_ascii import draw_ascii
    return draw_ascii({node.id: node_data_str(node) for node in self.nodes.values()}, self.edges)