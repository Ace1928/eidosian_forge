from swiftclient.utils import prt_bytes, split_request_headers
def print_container_stats(items, headers, output_manager):
    items.extend(headers_to_items(headers, meta_prefix='x-container-meta-', exclude_headers=('content-length', 'date', 'x-container-object-count', 'x-container-bytes-used', 'x-container-read', 'x-container-write', 'x-container-sync-to', 'x-container-sync-key')))
    offset = max((len(item) for item, value in items))
    output_manager.print_items(items, offset=offset)