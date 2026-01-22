import json
import re
from .widgets import Widget, DOMWidget, widget as widget_module
from .widgets.widget_link import Link
from .widgets.docutils import doc_subst
from ._version import __html_manager_version__
@doc_subst(_doc_snippets)
def embed_minimal_html(fp, views, title='IPyWidget export', template=None, **kwargs):
    """Write a minimal HTML file with widget views embedded.

    Parameters
    ----------
    fp: filename or file-like object
        The file to write the HTML output to.
    {views_attribute}
    title: title of the html page.
    template: Template in which to embed the widget state.
        This should be a Python string with placeholders
        `{{title}}` and `{{snippet}}`. The `{{snippet}}` placeholder
        will be replaced by all the widgets.
    {embed_kwargs}
    """
    snippet = embed_snippet(views, **kwargs)
    values = {'title': title, 'snippet': snippet}
    if template is None:
        template = html_template
    html_code = template.format(**values)
    if hasattr(fp, 'write'):
        fp.write(html_code)
    else:
        with open(fp, 'w') as f:
            f.write(html_code)