from __future__ import absolute_import, print_function, division
import operator
from petl.compat import string_types, izip
from petl.errors import ArgumentError
from petl.util.base import Table, dicts
def searchtextindex(index_or_dirname, query, limit=10, indexname=None, docnum_field=None, score_field=None, fieldboosts=None, search_kwargs=None):
    """
    Search a Whoosh index using a query. E.g.::

        >>> import petl as etl
        >>> import os
        >>> # set up an index and load some documents via the Whoosh API
        ... from whoosh.index import create_in
        >>> from whoosh.fields import *
        >>> schema = Schema(title=TEXT(stored=True), path=ID(stored=True),
        ...                            content=TEXT)
        >>> dirname = 'example.whoosh'
        >>> if not os.path.exists(dirname):
        ...     os.mkdir(dirname)
        ...
        >>> index = create_in('example.whoosh', schema)
        >>> writer = index.writer()
        >>> writer.add_document(title=u"Oranges", path=u"/a",
        ...                     content=u"This is the first document we've added!")
        >>> writer.add_document(title=u"Apples", path=u"/b",
        ...                     content=u"The second document is even more "
        ...                             u"interesting!")
        >>> writer.commit()
        >>> # demonstrate the use of searchtextindex()
        ... table1 = etl.searchtextindex('example.whoosh', 'oranges')
        >>> table1
        +------+-----------+
        | path | title     |
        +======+===========+
        | '/a' | 'Oranges' |
        +------+-----------+

        >>> table2 = etl.searchtextindex('example.whoosh', 'doc*')
        >>> table2
        +------+-----------+
        | path | title     |
        +======+===========+
        | '/a' | 'Oranges' |
        +------+-----------+
        | '/b' | 'Apples'  |
        +------+-----------+

    Keyword arguments:

    index_or_dirname
        Either an instance of `whoosh.index.Index` or a string containing the
        directory path where the index is to be stored.
    query
        Either a string or an instance of `whoosh.query.Query`. If a string,
        it will be parsed as a multi-field query, i.e., any terms not bound
        to a specific field will match **any** field.
    limit
        Return at most `limit` results.
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
    return SearchTextIndexView(index_or_dirname, query, limit=limit, indexname=indexname, docnum_field=docnum_field, score_field=score_field, fieldboosts=fieldboosts, search_kwargs=search_kwargs)