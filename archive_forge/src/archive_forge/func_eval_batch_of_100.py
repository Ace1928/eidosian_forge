import torch
from torch import nn
from torch import optim
from parlai.agents.transformer.modules import TransformerEncoder
from parlai.agents.transformer import transformer as Transformer
def eval_batch_of_100(self, context_encoded, captions_encoded):
    """
        Evaluate a batch of 100 examples.

        The captions of the other examples are used as negatives.

        :param context_encoded:
            the encoded context
        :param captions_encoded:
            the encoded captions

        :return:
            the total loss, the total number of correct examples, and the
            total number of examples evaluated.
        """
    total_loss = 0
    total_ok = 0
    num_examples = 0
    for i in range(0, len(context_encoded), 100):
        if i + 100 > len(context_encoded):
            break
        num_examples += 100
        loss, num_correct = self.evaluate_one_batch(context_encoded[i:i + 100, :], captions_encoded[i:i + 100, :])
        total_loss += loss.data.cpu().item()
        total_ok += num_correct.data.cpu().item()
    return (total_loss, total_ok, num_examples)