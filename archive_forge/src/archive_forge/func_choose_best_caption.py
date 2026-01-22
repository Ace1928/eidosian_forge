import torch
from torch import nn
from torch import optim
from parlai.agents.transformer.modules import TransformerEncoder
from parlai.agents.transformer import transformer as Transformer
def choose_best_caption(self, image_features, personalities, candidates, candidates_encoded=None, k=1):
    """
        Choose the best caption for each example.

        :param image_features:
            list of tensors of image features
        :param personalities:
            list of personalities
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
    context_encoded, _ = self.forward(image_features, personalities, None)
    context_encoded = context_encoded.detach()
    one_cand_set = True
    if candidates_encoded is None:
        one_cand_set = False
        candidates_encoded = [self.forward(None, None, c)[1].detach() for c in candidates]
    chosen = []
    for img_index in range(len(context_encoded)):
        context_encoding = context_encoded[img_index:img_index + 1, :]
        scores = torch.mm(candidates_encoded[img_index].to(context_encoding) if not one_cand_set else candidates_encoded.to(context_encoding), context_encoding.transpose(0, 1))
        if k >= 1:
            _, index_top = torch.topk(scores, k, dim=0)
        else:
            _, index_top = torch.topk(scores, scores.size(0), dim=0)
        chosen.append([candidates[img_index][idx] if not one_cand_set else candidates[idx] for idx in index_top.unsqueeze(1)])
    return chosen