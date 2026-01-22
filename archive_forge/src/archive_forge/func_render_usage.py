import csv
import functools
import logging
import re
from importlib import import_module
from humanfriendly.compat import StringIO
from humanfriendly.text import dedent, split_paragraphs, trim_empty_lines
def render_usage(text):
    """
    Reformat a command line program's usage message to reStructuredText_.

    :param text: The plain text usage message (a string).
    :returns: The usage message rendered to reStructuredText_ (a string).
    """
    meta_variables = find_meta_variables(text)
    introduction, options = parse_usage(text)
    output = [render_paragraph(p, meta_variables) for p in introduction]
    if options:
        output.append('\n'.join(['.. csv-table::', '   :header: Option, Description', '   :widths: 30, 70', '']))
        csv_buffer = StringIO()
        csv_writer = csv.writer(csv_buffer)
        while options:
            variants = options.pop(0)
            description = options.pop(0)
            csv_writer.writerow([render_paragraph(variants, meta_variables), '\n\n'.join((render_paragraph(p, meta_variables) for p in split_paragraphs(description))).rstrip()])
        csv_lines = csv_buffer.getvalue().splitlines()
        output.append('\n'.join(('   %s' % line for line in csv_lines)))
    logger.debug('Rendered output: %s', output)
    return '\n\n'.join((trim_empty_lines(o) for o in output))