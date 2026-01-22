from __future__ import absolute_import, print_function, division
import operator
from petl.compat import string_types, izip
from petl.errors import ArgumentError
from petl.util.base import Table, dicts
def searchtextindexpage(index_or_dirname, query, pagenum, pagelen=10, indexname=None, docnum_field=None, score_field=None, fieldboosts=None, search_kwargs=None):
    """
    Search an index using a query, returning a result page.

    Keyword arguments:

    index_or_dirname
        Either an instance of `whoosh.index.Index` or a string containing the
        directory path where the index is to be stored.
    query
        Either a string or an instance of `whoosh.query.Query`. If a string,
        it will be parsed as a multi-field query, i.e., any terms not bound
        to a specific field will match **any** field.
    pagenum
        Number of the page to return (e.g., 1 = first page).
    pagelen
        Number of results per page.
    indexname
        String containing the name of the index, if multiple indexes are stored
        in the same directory.
    docnum_field
        If not None, an extra field will be added to the output table containing
        the internal document number stored in the index. The name of the field
        will be the value of this argument.
    score_field
        If not None, an extra field will be added to the output table containing
        the score of the result. The name of the field will be the value of this
        argument.
    fieldboosts
        An optional dictionary mapping field names to boosts.
    search_kwargs
        Any extra keyword arguments to be passed through to the Whoosh
        `search()` method.

    """
    return SearchTextIndexView(index_or_dirname, query, pagenum=pagenum, pagelen=pagelen, indexname=indexname, docnum_field=docnum_field, score_field=score_field, fieldboosts=fieldboosts, search_kwargs=search_kwargs)