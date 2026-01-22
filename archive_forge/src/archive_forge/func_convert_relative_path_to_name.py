from __future__ import (absolute_import, division, print_function)
def convert_relative_path_to_name(path):
    """Calculate the module name from the given path.
        :type path: str
        :rtype: str
        """
    if path.endswith('/__init__.py'):
        clean_path = os.path.dirname(path)
    else:
        clean_path = path
    clean_path = os.path.splitext(clean_path)[0]
    name = clean_path.replace(os.path.sep, '.')
    if collection_loader:
        name = 'ansible_collections.%s.%s' % (collection_full_name, name)
    else:
        name = name[len('lib/'):]
    return name