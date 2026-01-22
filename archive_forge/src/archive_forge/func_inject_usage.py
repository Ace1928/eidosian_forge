import csv
import functools
import logging
import re
from importlib import import_module
from humanfriendly.compat import StringIO
from humanfriendly.text import dedent, split_paragraphs, trim_empty_lines
def inject_usage(module_name):
    """
    Use cog_ to inject a usage message into a reStructuredText_ file.

    :param module_name: The name of the module whose ``__doc__`` attribute is
                        the source of the usage message (a string).

    This simple wrapper around :func:`render_usage()` makes it very easy to
    inject a reformatted usage message into your documentation using cog_. To
    use it you add a fragment like the following to your ``*.rst`` file::

       .. [[[cog
       .. from humanfriendly.usage import inject_usage
       .. inject_usage('humanfriendly.cli')
       .. ]]]
       .. [[[end]]]

    The lines in the fragment above are single line reStructuredText_ comments
    that are not copied to the output. Their purpose is to instruct cog_ where
    to inject the reformatted usage message. Once you've added these lines to
    your ``*.rst`` file, updating the rendered usage message becomes really
    simple thanks to cog_:

    .. code-block:: sh

       $ cog.py -r README.rst

    This will inject or replace the rendered usage message in your
    ``README.rst`` file with an up to date copy.

    .. _cog: http://nedbatchelder.com/code/cog/
    """
    import cog
    usage_text = import_module(module_name).__doc__
    cog.out('\n' + render_usage(usage_text) + '\n\n')