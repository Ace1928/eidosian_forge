import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from parlai.core.torch_agent import TorchAgent, Output
from parlai.utils.misc import round_sigfigs, warn_once
from parlai.utils.torch import padded_tensor
from parlai.agents.transformer.transformer import TransformerRankerAgent
from .feedback_classifier.feedback_classifier import FeedbackClassifierRegex
from .modules import SelfFeedingModel
def satisfaction_step(self, batch):
    batchsize = batch.text_vec.size(0)
    if self.opt['regex']:
        contexts = [self.dict.vec2txt(vec) for vec in batch.text_vec]
        probs = self.rating_classifier.predict_proba(contexts).cuda()
    else:
        probs = self.model.score_satisfaction(batch.text_vec)
    preds = (probs > self.opt['rating_threshold']).long()
    if batch.labels is None:
        loss = None
    else:
        labels = torch.LongTensor([int(l) == 1 for l in batch.labels]).cuda()
        loss = self.satisfaction_criterion(probs, labels.float()).mean()
        self.update_sat_metrics(loss, preds, labels, batchsize)
    return (loss, preds)