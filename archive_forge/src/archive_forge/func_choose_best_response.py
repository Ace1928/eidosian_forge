import torch
from torch import nn
from parlai.agents.transformer.modules import (
from projects.personality_captions.transresnet.modules import (
def choose_best_response(self, image_features, personalities, dialogue_histories, candidates, candidates_encoded=None, k=1, batchsize=None):
    """
        Choose the best response for each example.

        :param image_features:
            list of tensors of image features
        :param personalities:
            list of personalities
        :param dialogue_histories:
            list of dialogue histories, one per example
        :param candidates:
            list of candidates, one set per example
        :param candidates_encoded:
            optional; if specified, a fixed set of encoded candidates that is
            used for each example
        :param k:
            number of ranked candidates to return. if < 1, we return the ranks
            of all candidates in the set.

        :return:
            a set of ranked candidates for each example
        """
    self.eval()
    _, _, encoded = self.forward(image_features, personalities, dialogue_histories, None, batchsize=batchsize)
    encoded = encoded.detach()
    one_cand_set = True
    if candidates_encoded is None:
        one_cand_set = False
        candidates_encoded = [self.forward_text_encoder(c).detach() for c in candidates]
    chosen = [self.choose_topk(idx if not one_cand_set else 0, encoded, candidates, candidates_encoded, one_cand_set, k) for idx in range(len(encoded))]
    return chosen