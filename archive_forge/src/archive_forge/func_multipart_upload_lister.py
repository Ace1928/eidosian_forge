from boto.compat import unquote_str
def multipart_upload_lister(bucket, key_marker='', upload_id_marker='', headers=None, encoding_type=None):
    """
    A generator function for listing multipart uploads in a bucket.
    """
    more_results = True
    k = None
    while more_results:
        rs = bucket.get_all_multipart_uploads(key_marker=key_marker, upload_id_marker=upload_id_marker, headers=headers, encoding_type=encoding_type)
        for k in rs:
            yield k
        key_marker = rs.next_key_marker
        if key_marker and encoding_type == 'url':
            key_marker = unquote_str(key_marker)
        upload_id_marker = rs.next_upload_id_marker
        more_results = rs.is_truncated