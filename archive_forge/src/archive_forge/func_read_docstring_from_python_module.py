from __future__ import (absolute_import, division, print_function)
import ast
import tokenize
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils.common.text.converters import to_text, to_native
from ansible.parsing.yaml.loader import AnsibleLoader
from ansible.utils.display import Display
def read_docstring_from_python_module(filename, verbose=True, ignore_errors=True):
    """
    Use tokenization to search for assignment of the documentation variables in the given file.
    Parse from YAML and return the resulting python structure or None together with examples as plain text.
    """
    seen = set()
    data = _init_doc_dict()
    next_string = None
    with tokenize.open(filename) as f:
        tokens = tokenize.generate_tokens(f.readline)
        for token in tokens:
            if token.type == tokenize.NAME:
                if token.start == 1 and token.string in string_to_vars and (token.string not in seen):
                    next_string = string_to_vars[token.string]
                    continue
            if next_string is not None and token.type == tokenize.STRING:
                seen.add(token.string)
                value = token.string
                if value.startswith(('r', 'b')):
                    value = value.lstrip('rb')
                if value.startswith(("'", '"')):
                    value = value.strip('\'"')
                if next_string == 'plainexamples':
                    data[next_string] = to_text(value)
                else:
                    try:
                        data[next_string] = AnsibleLoader(value, file_name=filename).get_single_data()
                    except Exception as e:
                        msg = "Unable to parse docs '%s' in python file '%s': %s" % (_var2string(next_string), filename, to_native(e))
                        if not ignore_errors:
                            raise AnsibleParserError(msg, orig_exc=e)
                        elif verbose:
                            display.error(msg)
                next_string = None
    if not seen:
        data = read_docstring_from_python_file(filename, verbose, ignore_errors)
    return data