import time
import logging
import mxnet as mx
from mxnet.module import Module
from .svrg_optimizer import _SVRGOptimizer
def update_full_grads(self, train_data):
    """Computes the gradients over all data w.r.t weights of past
        m epochs. For distributed env, it will accumulate full grads in the kvstore.

        Parameters
        ----------
        train_data: DataIter
            Train data iterator
        """
    param_names = self._exec_group.param_names
    arg, aux = self.get_params()
    self._mod_aux.set_params(arg_params=arg, aux_params=aux)
    train_data.reset()
    nbatch = 0
    padding = 0
    for batch in train_data:
        self._mod_aux.forward(batch, is_train=True)
        self._mod_aux.backward()
        nbatch += 1
        for ctx in range(self._ctx_len):
            for index, name in enumerate(param_names):
                grads = self._mod_aux._exec_group.grad_arrays[index][ctx]
                self._param_dict[ctx][name] = mx.nd.broadcast_add(self._param_dict[ctx][name], grads, axis=0)
        padding = batch.pad
    true_num_batch = nbatch - padding / train_data.batch_size
    for name in param_names:
        grad_list = []
        for i in range(self._ctx_len):
            self._param_dict[i][name] /= true_num_batch
            grad_list.append(self._param_dict[i][name])
        if self._kvstore:
            self._accumulate_kvstore(name, grad_list)