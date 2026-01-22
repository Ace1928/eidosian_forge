import railroad
from pip._vendor import pyparsing
import typing
from typing import (
from jinja2 import Template
from io import StringIO
import inspect
def extract_into_diagram(self, el_id: int):
    """
        Used when we encounter the same token twice in the same tree. When this
        happens, we replace all instances of that token with a terminal, and
        create a new subdiagram for the token
        """
    position = self[el_id]
    if position.parent:
        ret = EditablePartial.from_call(railroad.NonTerminal, text=position.name)
        if 'item' in position.parent.kwargs:
            position.parent.kwargs['item'] = ret
        elif 'items' in position.parent.kwargs:
            position.parent.kwargs['items'][position.parent_index] = ret
    if position.converted.func == railroad.Group:
        content = position.converted.kwargs['item']
    else:
        content = position.converted
    self.diagrams[el_id] = EditablePartial.from_call(NamedDiagram, name=position.name, diagram=EditablePartial.from_call(railroad.Diagram, content, **self.diagram_kwargs), index=position.number)
    del self[el_id]