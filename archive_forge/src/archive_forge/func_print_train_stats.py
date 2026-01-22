from collections import Counter, defaultdict
from nltk import jsontags
from nltk.tag import TaggerI
from nltk.tbl import Feature, Template
def print_train_stats():
    print('TEMPLATE STATISTICS (TRAIN)  {} templates, {} rules)'.format(len(template_counts), len(tids)))
    print('TRAIN ({tokencount:7d} tokens) initial {initialerrors:5d} {initialacc:.4f} final: {finalerrors:5d} {finalacc:.4f}'.format(**train_stats))
    head = '#ID | Score (train) |  #Rules     | Template'
    print(head, '\n', '-' * len(head), sep='')
    train_tplscores = sorted(weighted_traincounts.items(), key=det_tplsort, reverse=True)
    for tid, trainscore in train_tplscores:
        s = '{} | {:5d}   {:5.3f} |{:4d}   {:.3f} | {}'.format(tid, trainscore, trainscore / tottrainscores, template_counts[tid], template_counts[tid] / len(tids), Template.ALLTEMPLATES[int(tid)])
        print(s)