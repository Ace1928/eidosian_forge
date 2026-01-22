import html
import re
from collections import defaultdict
def ne_chunked():
    print()
    print('1500 Sentences from Penn Treebank, as processed by NLTK NE Chunker')
    print('=' * 45)
    ROLE = re.compile('.*(chairman|president|trader|scientist|economist|analyst|partner).*')
    rels = []
    for i, sent in enumerate(nltk.corpus.treebank.tagged_sents()[:1500]):
        sent = nltk.ne_chunk(sent)
        rels = extract_rels('PER', 'ORG', sent, corpus='ace', pattern=ROLE, window=7)
        for rel in rels:
            print(f'{i:<5}{rtuple(rel)}')