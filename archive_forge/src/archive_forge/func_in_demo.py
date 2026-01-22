import html
import re
from collections import defaultdict
def in_demo(trace=0, sql=True):
    """
    Select pairs of organizations and locations whose mentions occur with an
    intervening occurrence of the preposition "in".

    If the sql parameter is set to True, then the entity pairs are loaded into
    an in-memory database, and subsequently pulled out using an SQL "SELECT"
    query.
    """
    from nltk.corpus import ieer
    if sql:
        try:
            import sqlite3
            connection = sqlite3.connect(':memory:')
            cur = connection.cursor()
            cur.execute('create table Locations\n            (OrgName text, LocationName text, DocID text)')
        except ImportError:
            import warnings
            warnings.warn('Cannot import sqlite; sql flag will be ignored.')
    IN = re.compile('.*\\bin\\b(?!\\b.+ing)')
    print()
    print('IEER: in(ORG, LOC) -- just the clauses:')
    print('=' * 45)
    for file in ieer.fileids():
        for doc in ieer.parsed_docs(file):
            if trace:
                print(doc.docno)
                print('=' * 15)
            for rel in extract_rels('ORG', 'LOC', doc, corpus='ieer', pattern=IN):
                print(clause(rel, relsym='IN'))
                if sql:
                    try:
                        rtuple = (rel['subjtext'], rel['objtext'], doc.docno)
                        cur.execute('insert into Locations\n                                    values (?, ?, ?)', rtuple)
                        connection.commit()
                    except NameError:
                        pass
    if sql:
        try:
            cur.execute("select OrgName from Locations\n                        where LocationName = 'Atlanta'")
            print()
            print('Extract data from SQL table: ORGs in Atlanta')
            print('-' * 15)
            for row in cur:
                print(row)
        except NameError:
            pass