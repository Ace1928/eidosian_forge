import random
import io
import os
import pickle
from parlai.utils.misc import msg_to_str
def build_from_db(opt, db_path, data_path, fname, fname2):
    dbp = os.path.join(db_path, fname)
    file = open(dbp, 'rb')
    db = pickle.load(file)
    dbp = os.path.join(db_path, fname2)
    file = open(dbp, 'rb')
    db_unseen = pickle.load(file)
    for i in range(0, len(db)):
        db[i]['split'] = 'train'
    for i in range(0, len(db_unseen)):
        db_unseen[i]['split'] = 'test_unseen'
    rand2 = random.Random(42)
    x = []
    for i in range(1368, 4733):
        x.append(i)
    rand2.shuffle(x)
    for i in range(0, 1000):
        db[x[i]]['split'] = 'test'
    for i in range(1000, 1500):
        db[x[i]]['split'] = 'valid'
    for split in ['train', 'valid', 'test']:
        write_out_candidates(db, data_path, split)
        write_alldata(opt, db, data_path, 'speech', split)
        write_alldata(opt, db, data_path, 'action', split)
        write_alldata(opt, db, data_path, 'emote', split)
        write_alldata(opt, db, data_path, 'which', split)
    for split in ['test_unseen']:
        write_out_candidates(db_unseen, data_path, split)
        write_alldata(opt, db_unseen, data_path, 'speech', split)
        write_alldata(opt, db_unseen, data_path, 'action', split)
        write_alldata(opt, db_unseen, data_path, 'emote', split)
        write_alldata(opt, db_unseen, data_path, 'which', split)