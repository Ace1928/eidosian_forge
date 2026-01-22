from boto.compat import six
def query_lister(domain, query='', max_items=None, attr_names=None):
    more_results = True
    num_results = 0
    next_token = None
    while more_results:
        rs = domain.connection.query_with_attributes(domain, query, attr_names, next_token=next_token)
        for item in rs:
            if max_items:
                if num_results == max_items:
                    raise StopIteration
            yield item
            num_results += 1
        next_token = rs.next_token
        more_results = next_token is not None