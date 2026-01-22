from parlai.core.dict import DictionaryAgent
from abc import ABC, abstractmethod
def override_special_tokens(self, opt):
    self._define_special_tokens(opt)
    self.start_idx = self.tokenizer.convert_tokens_to_ids([self.start_token])[0]
    self.end_idx = self.tokenizer.convert_tokens_to_ids([self.end_token])[0]
    self.null_idx = self.tokenizer.convert_tokens_to_ids([self.null_token])[0]
    self.tok2ind[self.end_token] = self.end_idx
    self.tok2ind[self.start_token] = self.start_idx
    self.tok2ind[self.null_token] = self.null_idx
    self.ind2tok[self.end_idx] = self.end_token
    self.ind2tok[self.start_idx] = self.start_token
    self.ind2tok[self.null_idx] = self.null_token