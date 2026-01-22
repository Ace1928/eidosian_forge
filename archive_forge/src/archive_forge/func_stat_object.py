from swiftclient.utils import prt_bytes, split_request_headers
def stat_object(conn, options, container, obj):
    req_headers = split_request_headers(options.get('header', []))
    query_string = None
    if options.get('version_id') is not None:
        query_string = 'version-id=%s' % options['version_id']
    headers = conn.head_object(container, obj, headers=req_headers, query_string=query_string)
    items = []
    if options['verbose'] > 1:
        path = '%s/%s/%s' % (conn.url, container, obj)
        items.extend([('URL', path), ('Auth Token', conn.token)])
    content_length = prt_bytes(headers.get('content-length', 0), options['human']).lstrip()
    items.extend([('Account', conn.url.rsplit('/', 1)[-1]), ('Container', container), ('Object', obj), ('Content Type', headers.get('content-type')), ('Content Length', content_length), ('Last Modified', headers.get('last-modified')), ('ETag', headers.get('etag')), ('Manifest', headers.get('x-object-manifest'))])
    return (items, headers)