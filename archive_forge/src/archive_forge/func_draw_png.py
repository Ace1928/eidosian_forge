from __future__ import annotations
import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import (
from uuid import UUID, uuid4
from langchain_core.pydantic_v1 import BaseModel
def draw_png(self, output_file_path: Optional[str]=None, fontname: Optional[str]=None, labels: Optional[LabelsDict]=None) -> Union[bytes, None]:
    from langchain_core.runnables.graph_png import PngDrawer
    default_node_labels = {node.id: node_data_str(node) for node in self.nodes.values()}
    return PngDrawer(fontname, LabelsDict(nodes={**default_node_labels, **(labels['nodes'] if labels is not None else {})}, edges=labels['edges'] if labels is not None else {})).draw(self, output_file_path)