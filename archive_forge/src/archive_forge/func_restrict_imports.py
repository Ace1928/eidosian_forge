from __future__ import (absolute_import, division, print_function)
@contextlib.contextmanager
def restrict_imports(path, name, messages, restrict_to_module_paths):
    """Restrict available imports.
        :type path: str
        :type name: str
        :type messages: set[str]
        :type restrict_to_module_paths: bool
        """
    restricted_loader = RestrictedModuleLoader(path, name, restrict_to_module_paths)
    sys.meta_path.insert(0, restricted_loader)
    sys.path_importer_cache.clear()
    try:
        yield
    finally:
        if import_type == 'plugin' and (not collection_loader):
            from ansible.utils.collection_loader._collection_finder import _AnsibleCollectionFinder
            _AnsibleCollectionFinder._remove()
        if sys.meta_path[0] != restricted_loader:
            report_message(path, 0, 0, 'metapath', 'changes to sys.meta_path[0] are not permitted', messages)
        while restricted_loader in sys.meta_path:
            sys.meta_path.remove(restricted_loader)
        sys.path_importer_cache.clear()