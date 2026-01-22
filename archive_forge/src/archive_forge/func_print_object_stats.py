from swiftclient.utils import prt_bytes, split_request_headers
def print_object_stats(items, headers, output_manager):
    items.extend(headers_to_items(headers, meta_prefix='x-object-meta-', exclude_headers=('content-type', 'content-length', 'last-modified', 'etag', 'date', 'x-object-manifest')))
    offset = max((len(item) for item, value in items))
    output_manager.print_items(items, offset=offset, skip_missing=True)