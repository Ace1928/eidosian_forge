import torch
from torch import nn
from torch import optim
from parlai.agents.transformer.modules import TransformerEncoder
from parlai.agents.transformer import transformer as Transformer
def train_batch(self, image_features, personalities, captions):
    """
        Batch train on a set of examples.

        Uses captions from other examples as negatives during training

        :param image_features:
            list of tensors of image features
        :param personalities:
            list of personalities
        :param captions:
            list of captions

        :return:
            the total loss, the number of correct examples, and the number of
            examples trained on
        """
    self.zero_grad()
    self.train()
    context_encoded, captions_encoded = self.forward(image_features, personalities, captions)
    loss, num_correct = self.evaluate_one_batch(context_encoded, captions_encoded, during_train=True)
    loss.backward()
    self.optimizer.step()
    loss, num_correct, num_examples = self.eval_batch_of_100(context_encoded, captions_encoded)
    return (loss, num_correct, num_examples)