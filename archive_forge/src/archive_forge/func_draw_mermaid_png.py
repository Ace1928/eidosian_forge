from __future__ import annotations
import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import (
from uuid import UUID, uuid4
from langchain_core.pydantic_v1 import BaseModel
def draw_mermaid_png(self, *, curve_style: CurveStyle=CurveStyle.LINEAR, node_colors: NodeColors=NodeColors(start='#ffdfba', end='#baffc9', other='#fad7de'), wrap_label_n_words: int=9, output_file_path: Optional[str]=None, draw_method: MermaidDrawMethod=MermaidDrawMethod.API, background_color: str='white', padding: int=10) -> bytes:
    from langchain_core.runnables.graph_mermaid import draw_mermaid_png
    mermaid_syntax = self.draw_mermaid(curve_style=curve_style, node_colors=node_colors, wrap_label_n_words=wrap_label_n_words)
    return draw_mermaid_png(mermaid_syntax=mermaid_syntax, output_file_path=output_file_path, draw_method=draw_method, background_color=background_color, padding=padding)