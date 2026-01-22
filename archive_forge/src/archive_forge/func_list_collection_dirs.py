from ansible.errors import AnsibleError
from ansible.cli.galaxy import with_collection_artifacts_manager
from ansible.galaxy.collection import find_existing_collections
from ansible.module_utils.common.text.converters import to_bytes
from ansible.utils.collection_loader._collection_finder import _get_collection_name_from_path
from ansible.utils.display import Display
@with_collection_artifacts_manager
def list_collection_dirs(search_paths=None, coll_filter=None, artifacts_manager=None, dedupe=True):
    """
    Return paths for the specific collections found in passed or configured search paths
    :param search_paths: list of text-string paths, if none load default config
    :param coll_filter: limit collections to just the specific namespace or collection, if None all are returned
    :return: list of collection directory paths
    """
    namespace_filter = None
    collection_filter = None
    has_pure_namespace_filter = False
    if coll_filter is not None:
        if isinstance(coll_filter, str):
            coll_filter = [coll_filter]
        namespace_filter = set()
        for coll_name in coll_filter:
            if '.' in coll_name:
                try:
                    namespace, collection = coll_name.split('.')
                except ValueError:
                    raise AnsibleError('Invalid collection pattern supplied: %s' % coll_name)
                namespace_filter.add(namespace)
                if not has_pure_namespace_filter:
                    if collection_filter is None:
                        collection_filter = []
                    collection_filter.append(collection)
            else:
                namespace_filter.add(coll_name)
                has_pure_namespace_filter = True
                collection_filter = None
        namespace_filter = sorted(namespace_filter)
    for req in find_existing_collections(search_paths, artifacts_manager, namespace_filter=namespace_filter, collection_filter=collection_filter, dedupe=dedupe):
        if not has_pure_namespace_filter and coll_filter is not None and (req.fqcn not in coll_filter):
            continue
        yield to_bytes(req.src)