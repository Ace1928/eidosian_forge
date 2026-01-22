from __future__ import annotations
import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import (
from uuid import UUID, uuid4
from langchain_core.pydantic_v1 import BaseModel
def trim_last_node(self) -> None:
    """Remove the last node if it exists and has a single incoming edge,
        ie. if removing it would not leave the graph without a "last" node."""
    last_node = self.last_node()
    if last_node:
        if len(self.nodes) == 1 or len([edge for edge in self.edges if edge.target == last_node.id]) == 1:
            self.remove_node(last_node)