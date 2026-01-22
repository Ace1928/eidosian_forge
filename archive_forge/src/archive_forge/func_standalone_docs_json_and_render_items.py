from __future__ import annotations
import logging # isort:skip
import re
from contextlib import contextmanager
from typing import (
from weakref import WeakKeyDictionary
from ..core.types import ID
from ..document.document import Document
from ..model import Model, collect_models
from ..settings import settings
from ..themes.theme import Theme
from ..util.dataclasses import dataclass, field
from ..util.serialization import (
def standalone_docs_json_and_render_items(models: Model | Document | Sequence[Model | Document], *, suppress_callback_warning: bool=False) -> tuple[dict[ID, DocJson], list[RenderItem]]:
    """

    """
    if isinstance(models, (Model, Document)):
        models = [models]
    if not (isinstance(models, Sequence) and all((isinstance(x, (Model, Document)) for x in models))):
        raise ValueError('Expected a Model, Document, or Sequence of Models or Documents')
    if submodel_has_python_callbacks(models) and (not suppress_callback_warning):
        log.warning(_CALLBACKS_WARNING)
    docs: dict[Document, tuple[ID, dict[Model, ID]]] = {}
    for model_or_doc in models:
        if isinstance(model_or_doc, Document):
            model = None
            doc = model_or_doc
        else:
            model = model_or_doc
            doc = model.document
            if doc is None:
                raise ValueError('A Bokeh Model must be part of a Document to render as standalone content')
        if doc not in docs:
            docs[doc] = (make_globally_unique_id(), dict())
        docid, roots = docs[doc]
        if model is not None:
            roots[model] = make_globally_unique_css_safe_id()
        else:
            for model in doc.roots:
                roots[model] = make_globally_unique_css_safe_id()
    docs_json: dict[ID, DocJson] = {}
    for doc, (docid, _) in docs.items():
        docs_json[docid] = doc.to_json(deferred=False)
    render_items: list[RenderItem] = []
    for _, (docid, roots) in docs.items():
        render_items.append(RenderItem(docid, roots=roots))
    return (docs_json, render_items)