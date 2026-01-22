from swiftclient.utils import prt_bytes, split_request_headers
def stat_container(conn, options, container):
    req_headers = split_request_headers(options.get('header', []))
    headers = conn.head_container(container, headers=req_headers)
    items = []
    if options['verbose'] > 1:
        path = '%s/%s' % (conn.url, container)
        items.extend([('URL', path), ('Auth Token', conn.token)])
    object_count = prt_bytes(headers.get('x-container-object-count', 0), options['human']).lstrip()
    bytes_used = prt_bytes(headers.get('x-container-bytes-used', 0), options['human']).lstrip()
    items.extend([('Account', conn.url.rsplit('/', 1)[-1]), ('Container', container), ('Objects', object_count), ('Bytes', bytes_used), ('Read ACL', headers.get('x-container-read', '')), ('Write ACL', headers.get('x-container-write', '')), ('Sync To', headers.get('x-container-sync-to', '')), ('Sync Key', headers.get('x-container-sync-key', ''))])
    return (items, headers)