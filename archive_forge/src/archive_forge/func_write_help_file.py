import os
import sys
import shutil
import importlib
import textwrap
import re
import warnings
from ._all_keywords import r_keywords
from ._py_components_generation import reorder_props
def write_help_file(name, props, description, prefix, rpkg_data):
    """Write R documentation file (.Rd) given component name and properties.

    Parameters
    ----------
    name = the name of the Dash component for which a help file is generated
    props = the properties of the component
    description = the component's description, inserted into help file header
    prefix = the DashR library prefix (optional, can be a blank string)
    rpkg_data = package metadata (optional)

    Returns
    -------
    writes an R help file to the man directory for the generated R package
    """
    funcname = format_fn_name(prefix, name)
    file_name = funcname + '.Rd'
    wildcards = ''
    default_argtext = ''
    item_text = ''
    accepted_wildcards = ''
    value_text = 'named list of JSON elements corresponding to React.js properties and their values'
    prop_keys = list(props.keys())
    if any((key.endswith('-*') for key in prop_keys)):
        accepted_wildcards = get_wildcards_r(prop_keys)
        wildcards = ', ...'
    for item in prop_keys[:]:
        if item.endswith('-*') or item in r_keywords or item == 'setProps':
            prop_keys.remove(item)
    default_argtext += ', '.join(('{}=NULL'.format(p) for p in prop_keys))
    item_text += '\n\n'.join(('\\item{{{}}}{{{}{}}}'.format(p, print_r_type(props[p]['type']), props[p]['description']) for p in prop_keys))
    description = re.sub('(?<!\\\\)%', '\\%', description)
    item_text = re.sub('(?<!\\\\)%', '\\%', item_text)
    if '**Example Usage**' in description:
        description = description.split('**Example Usage**')[0].rstrip()
    if wildcards == ', ...':
        default_argtext += wildcards
        item_text += wildcard_help_template.format(accepted_wildcards)
    file_path = os.path.join('man', file_name)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(help_string.format(funcname=funcname, name=name, default_argtext=textwrap.fill(default_argtext, width=60, break_long_words=False), item_text=item_text, value_text=value_text, description=description.replace('\n', ' ')))
    if rpkg_data is not None and 'r_examples' in rpkg_data:
        ex = rpkg_data.get('r_examples')
        the_ex = ([e for e in ex if e.get('name') == funcname] or [None])[0]
        result = ''
        if the_ex and 'code' in the_ex.keys():
            result += wrap('examples', wrap('dontrun' if the_ex.get('dontrun') else '', the_ex['code']))
            with open(file_path, 'a+', encoding='utf-8') as fa:
                fa.write(result + '\n')