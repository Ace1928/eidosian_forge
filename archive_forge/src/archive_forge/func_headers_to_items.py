from swiftclient.utils import prt_bytes, split_request_headers
def headers_to_items(headers, meta_prefix='', exclude_headers=None):
    exclude_headers = exclude_headers or []
    other_items = []
    meta_items = []
    for key, value in headers.items():
        if key not in exclude_headers:
            if key.startswith(meta_prefix):
                meta_key = 'Meta %s' % key[len(meta_prefix):].title()
                meta_items.append((meta_key, value))
            else:
                other_items.append((key.title(), value))
    return meta_items + other_items