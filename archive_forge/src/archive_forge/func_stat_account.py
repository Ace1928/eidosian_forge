from swiftclient.utils import prt_bytes, split_request_headers
def stat_account(conn, options):
    items = []
    req_headers = split_request_headers(options.get('header', []))
    headers = conn.head_account(headers=req_headers)
    if options['verbose'] > 1:
        items.extend([('StorageURL', conn.url), ('Auth Token', conn.token)])
    container_count = int(headers.get('x-account-container-count', 0))
    object_count = prt_bytes(headers.get('x-account-object-count', 0), options['human']).lstrip()
    bytes_used = prt_bytes(headers.get('x-account-bytes-used', 0), options['human']).lstrip()
    items.extend([('Account', conn.url.rsplit('/', 1)[-1]), ('Containers', container_count), ('Objects', object_count), ('Bytes', bytes_used)])
    if headers.get('x-account-meta-quota-bytes'):
        quota_bytes = prt_bytes(headers.get('x-account-meta-quota-bytes'), options['human']).lstrip()
        items.append(('Quota Bytes', quota_bytes))
    policies = set()
    for header_key, header_value in headers.items():
        if header_key.lower().startswith(POLICY_HEADER_PREFIX):
            policy_name = header_key.rsplit('-', 2)[0].split('-', 4)[-1]
            policies.add(policy_name)
    for policy in policies:
        container_count_header = POLICY_HEADER_PREFIX + policy + '-container-count'
        if container_count_header in headers:
            items.append(('Containers in policy "' + policy + '"', prt_bytes(headers[container_count_header], options['human']).lstrip()))
        items.extend((('Objects in policy "' + policy + '"', prt_bytes(headers.get(POLICY_HEADER_PREFIX + policy + '-object-count', 0), options['human']).lstrip()), ('Bytes in policy "' + policy + '"', prt_bytes(headers.get(POLICY_HEADER_PREFIX + policy + '-bytes-used', 0), options['human']).lstrip())))
        policy_quota = headers.get(PER_POLICY_QUOTA_HEADER_PREFIX + policy)
        if policy_quota:
            items.append(('Quota Bytes for policy "' + policy + '"', prt_bytes(policy_quota, options['human']).lstrip()))
    return (items, headers)