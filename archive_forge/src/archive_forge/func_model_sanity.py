from __future__ import with_statement, division
import logging
import unittest
import os
from collections import namedtuple
import numpy as np
from testfixtures import log_capture
from gensim import utils
from gensim.models import doc2vec, keyedvectors
from gensim.test.utils import datapath, get_tmpfile, temporary_file, common_texts as raw_sentences
def model_sanity(self, model, keep_training=True):
    """Any non-trivial model on DocsLeeCorpus can pass these sanity checks"""
    fire1 = 0
    fire2 = np.int64(8)
    alt1 = 29
    doc0_inferred = model.infer_vector(list(DocsLeeCorpus())[0].words)
    sims_to_infer = model.dv.most_similar([doc0_inferred], topn=len(model.dv))
    sims_ids = [docid for docid, sim in sims_to_infer]
    self.assertTrue(fire1 in sims_ids, '{0} not found in {1}'.format(fire1, sims_to_infer))
    f_rank = sims_ids.index(fire1)
    self.assertLess(f_rank, 10)
    sims = model.dv.most_similar(fire1, topn=len(model.dv))
    f2_rank = [docid for docid, sim in sims].index(fire2)
    self.assertLess(f2_rank, 30)
    doc0_vec = model.dv[fire1]
    sims2 = model.dv.most_similar(positive=[doc0_vec], topn=21)
    sims2 = [(id, sim) for id, sim in sims2 if id != fire1]
    sims = sims[:20]
    self.assertEqual(list(zip(*sims))[0], list(zip(*sims2))[0])
    self.assertTrue(np.allclose(list(zip(*sims))[1], list(zip(*sims2))[1]))
    clip_sims = model.dv.most_similar(fire1, clip_start=len(model.dv) // 2, clip_end=len(model.dv) * 2 // 3)
    sims_doc_id = [docid for docid, sim in clip_sims]
    for s_id in sims_doc_id:
        self.assertTrue(len(model.dv) // 2 <= s_id <= len(model.dv) * 2 // 3)
    self.assertLess(model.dv.similarity(fire1, alt1), model.dv.similarity(fire1, fire2))
    self.assertLess(model.dv.similarity(fire2, alt1), model.dv.similarity(fire1, fire2))
    self.assertEqual(model.dv.doesnt_match([fire1, alt1, fire2]), alt1)
    if keep_training:
        tmpf = get_tmpfile('gensim_doc2vec_resave.tst')
        model.save(tmpf)
        loaded = doc2vec.Doc2Vec.load(tmpf)
        loaded.train(corpus_iterable=sentences, total_examples=loaded.corpus_count, epochs=loaded.epochs)