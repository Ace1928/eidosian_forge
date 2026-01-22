from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible_collections.ansible.utils.plugins.plugin_utils.base.cli_parser import CliParserBase
from ansible_collections.ansible.netcommon.plugins.module_utils.cli_parser.cli_parsertemplate import (
Std entry point for a cli_parse parse execution

        :return: Errors or parsed text as structured data
        :rtype: dict

        :example:

        The parse function of a parser should return a dict:
        {"errors": [a list of errors]}
        or
        {"parsed": obj}
        