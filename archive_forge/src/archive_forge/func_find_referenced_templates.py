import typing as t
from . import nodes
from .compiler import CodeGenerator
from .compiler import Frame
def find_referenced_templates(ast: nodes.Template) -> t.Iterator[t.Optional[str]]:
    """Finds all the referenced templates from the AST.  This will return an
    iterator over all the hardcoded template extensions, inclusions and
    imports.  If dynamic inheritance or inclusion is used, `None` will be
    yielded.

    >>> from jinja2 import Environment, meta
    >>> env = Environment()
    >>> ast = env.parse('{% extends "layout.html" %}{% include helper %}')
    >>> list(meta.find_referenced_templates(ast))
    ['layout.html', None]

    This function is useful for dependency tracking.  For example if you want
    to rebuild parts of the website after a layout template has changed.
    """
    template_name: t.Any
    for node in ast.find_all(_ref_types):
        template: nodes.Expr = node.template
        if not isinstance(template, nodes.Const):
            if isinstance(template, (nodes.Tuple, nodes.List)):
                for template_name in template.items:
                    if isinstance(template_name, nodes.Const):
                        if isinstance(template_name.value, str):
                            yield template_name.value
                    else:
                        yield None
            else:
                yield None
            continue
        if isinstance(template.value, str):
            yield template.value
        elif isinstance(node, nodes.Include) and isinstance(template.value, (tuple, list)):
            for template_name in template.value:
                if isinstance(template_name, str):
                    yield template_name
        else:
            yield None