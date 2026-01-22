import pytest
from thinc.api import (
from thinc.compat import has_tensorflow, has_torch
@pytest.mark.slow
@pytest.mark.parametrize(('width', 'nb_epoch', 'min_score'), [(32, 20, 0.8)])
def test_small_end_to_end(width, nb_epoch, min_score, create_model, mnist):
    batch_size = 128
    dropout = 0.2
    (train_X, train_Y), (dev_X, dev_Y) = mnist
    model = create_model(width, dropout, nI=train_X.shape[1], nO=train_Y.shape[1])
    model.initialize(X=train_X[:5], Y=train_Y[:5])
    optimizer = Adam(0.001)
    losses = []
    scores = []
    ops = get_current_ops()
    for i in range(nb_epoch):
        for X, Y in model.ops.multibatch(batch_size, train_X, train_Y, shuffle=True):
            Yh, backprop = model.begin_update(X)
            Yh = ops.asarray(Yh)
            backprop(Yh - Y)
            model.finish_update(optimizer)
            losses.append(((Yh - Y) ** 2).sum())
        correct = 0
        total = 0
        for X, Y in model.ops.multibatch(batch_size, dev_X, dev_Y):
            Yh = model.predict(X)
            Yh = ops.asarray(Yh)
            correct += (Yh.argmax(axis=1) == Y.argmax(axis=1)).sum()
            total += Yh.shape[0]
        score = correct / total
        scores.append(score)
    assert losses[-1] < losses[0], losses
    if scores[0] < 1.0:
        assert scores[-1] > scores[0], scores
    assert any([score > min_score for score in scores]), scores