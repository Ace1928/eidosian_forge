import railroad
from pip._vendor import pyparsing
import typing
from typing import (
from jinja2 import Template
from io import StringIO
import inspect
def railroad_to_html(diagrams: List[NamedDiagram], embed=False, **kwargs) -> str:
    """
    Given a list of NamedDiagram, produce a single HTML string that visualises those diagrams
    :params kwargs: kwargs to be passed in to the template
    """
    data = []
    for diagram in diagrams:
        if diagram.diagram is None:
            continue
        io = StringIO()
        try:
            css = kwargs.get('css')
            diagram.diagram.writeStandalone(io.write, css=css)
        except AttributeError:
            diagram.diagram.writeSvg(io.write)
        title = diagram.name
        if diagram.index == 0:
            title += ' (root)'
        data.append({'title': title, 'text': '', 'svg': io.getvalue()})
    return template.render(diagrams=data, embed=embed, **kwargs)