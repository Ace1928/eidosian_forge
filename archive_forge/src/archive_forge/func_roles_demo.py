import html
import re
from collections import defaultdict
def roles_demo(trace=0):
    from nltk.corpus import ieer
    roles = '\n    (.*(                   # assorted roles\n    analyst|\n    chair(wo)?man|\n    commissioner|\n    counsel|\n    director|\n    economist|\n    editor|\n    executive|\n    foreman|\n    governor|\n    head|\n    lawyer|\n    leader|\n    librarian).*)|\n    manager|\n    partner|\n    president|\n    producer|\n    professor|\n    researcher|\n    spokes(wo)?man|\n    writer|\n    ,\\sof\\sthe?\\s*  # "X, of (the) Y"\n    '
    ROLES = re.compile(roles, re.VERBOSE)
    print()
    print('IEER: has_role(PER, ORG) -- raw rtuples:')
    print('=' * 45)
    for file in ieer.fileids():
        for doc in ieer.parsed_docs(file):
            lcon = rcon = False
            if trace:
                print(doc.docno)
                print('=' * 15)
                lcon = rcon = True
            for rel in extract_rels('PER', 'ORG', doc, corpus='ieer', pattern=ROLES):
                print(rtuple(rel, lcon=lcon, rcon=rcon))