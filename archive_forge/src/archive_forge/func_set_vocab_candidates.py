from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.utils.torch import padded_3d
from parlai.zoo.bert.build import download
from .bert_dictionary import BertDictionaryAgent
from .helpers import (
import os
import torch
from tqdm import tqdm
def set_vocab_candidates(self, shared):
    """
        Load the tokens from the vocab as candidates.

        self.vocab_candidates will contain a [num_cands] list of strings
        self.vocab_candidate_vecs will contain a [num_cands, 1] LongTensor
        """
    self.opt['encode_candidate_vecs'] = True
    if shared:
        self.vocab_candidates = shared['vocab_candidates']
        self.vocab_candidate_vecs = shared['vocab_candidate_vecs']
        self.vocab_candidate_encs = shared['vocab_candidate_encs']
    elif 'vocab' in (self.opt['candidates'], self.opt['eval_candidates']):
        cands = []
        vecs = []
        for ind in range(1, len(self.dict)):
            txt = self.dict[ind]
            cands.append(txt)
            vecs.append(self._vectorize_text(txt, add_start=True, add_end=True, truncate=self.label_truncate))
        self.vocab_candidates = cands
        self.vocab_candidate_vecs = padded_3d([vecs]).squeeze(0)
        print('[ Loaded fixed candidate set (n = {}) from vocabulary ]'.format(len(self.vocab_candidates)))
        enc_path = self.opt.get('model_file') + '.vocab.encs'
        if os.path.isfile(enc_path):
            self.vocab_candidate_encs = self.load_candidates(enc_path, cand_type='vocab encodings')
        else:
            cand_encs = []
            vec_batches = [self.vocab_candidate_vecs[i:i + 512] for i in range(0, len(self.vocab_candidate_vecs), 512)]
            print('[ Vectorizing vocab candidates ({} batch(es) of up to 512) ]'.format(len(vec_batches)))
            for vec_batch in tqdm(vec_batches):
                cand_encs.append(self.encode_candidates(vec_batch))
            self.vocab_candidate_encs = torch.cat(cand_encs, 0)
            self.save_candidates(self.vocab_candidate_encs, enc_path, cand_type='vocab encodings')
        if self.use_cuda:
            self.vocab_candidate_vecs = self.vocab_candidate_vecs.cuda()
            self.vocab_candidate_encs = self.vocab_candidate_encs.cuda()
    else:
        self.vocab_candidates = None
        self.vocab_candidate_vecs = None
        self.vocab_candidate_encs = None