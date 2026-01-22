from urllib import parse as urlparse
def get_collection_links(request, items):
    """Retrieve 'next' link, if applicable."""
    links = []
    try:
        limit = int(request.params.get('limit') or 0)
    except ValueError:
        limit = 0
    if limit > 0 and limit == len(items):
        last_item = items[-1]
        last_item_id = last_item['id']
        links.append({'rel': 'next', 'href': _get_next_link(request, last_item_id)})
    return links