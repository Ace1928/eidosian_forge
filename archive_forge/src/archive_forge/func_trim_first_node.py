from __future__ import annotations
import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import (
from uuid import UUID, uuid4
from langchain_core.pydantic_v1 import BaseModel
def trim_first_node(self) -> None:
    """Remove the first node if it exists and has a single outgoing edge,
        ie. if removing it would not leave the graph without a "first" node."""
    first_node = self.first_node()
    if first_node:
        if len(self.nodes) == 1 or len([edge for edge in self.edges if edge.source == first_node.id]) == 1:
            self.remove_node(first_node)