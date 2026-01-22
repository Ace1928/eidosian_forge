from __future__ import unicode_literals
from pybtex.plugin import find_plugin
from pybtex import database
def format_database(from_filename, to_filename, bib_format=None, output_backend=None, input_encoding=None, output_encoding=None, parser_options=None, min_crossrefs=2, style=None, **kwargs):
    if parser_options is None:
        parser_options = {}
    output_backend = find_plugin('pybtex.backends', output_backend, filename=to_filename)
    bib_data = database.parse_file(from_filename, encoding=input_encoding, bib_format=bib_format, **parser_options)
    style_cls = find_plugin('pybtex.style.formatting', style)
    style = style_cls(label_style=kwargs.get('label_style'), name_style=kwargs.get('name_style'), sorting_style=kwargs.get('sorting_style'), abbreviate_names=kwargs.get('abbreviate_names'), min_crossrefs=min_crossrefs)
    formatted_bibliography = style.format_bibliography(bib_data)
    output_backend(output_encoding).write_to_file(formatted_bibliography, to_filename)