from math import ceil
from boto.compat import json, map, six
import requests
def get_num_hits(self, query):
    """Return the total number of hits for query

        :type query: :class:`boto.cloudsearch.search.Query`
        :param query: a group of search criteria

        :rtype: int
        :return: Total number of hits for query
        """
    query.update_size(1)
    return self(query).hits