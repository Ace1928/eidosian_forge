from ...utils import split_and_load
from .... import autograd
def fit_batch(self, estimator, train_batch, batch_axis=0):
    """Trains the estimator model on a batch of training data.

        Parameters
        ----------
        estimator : Estimator
            Reference to the estimator
        train_batch : tuple
            Data and label of a batch from the training data loader.
        batch_axis : int, default 0
            Batch axis to split the training data into devices.

        Returns
        -------
        data: List of NDArray
            Sharded data from the batch. Data is sharded with
            `gluon.split_and_load`.
        label: List of NDArray
            Sharded label from the batch. Labels are sharded with
            `gluon.split_and_load`.
        pred: List of NDArray
            Prediction on each of the sharded inputs.
        loss: List of NDArray
            Loss on each of the sharded inputs.
        """
    data, label = self._get_data_and_label(train_batch, estimator.context, batch_axis)
    with autograd.record():
        pred = [estimator.net(x) for x in data]
        loss = [estimator.loss(y_hat, y) for y_hat, y in zip(pred, label)]
    for l in loss:
        l.backward()
    return (data, label, pred, loss)