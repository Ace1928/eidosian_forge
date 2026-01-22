import warnings
import functools
from .. import symbol, init, ndarray
from ..base import string_types, numeric_types
def unfuse(self):
    """Unfuse the fused RNN in to a stack of rnn cells.

        Returns
        -------
        cell : mxnet.rnn.SequentialRNNCell
            unfused cell that can be used for stepping, and can run on CPU.
        """
    stack = SequentialRNNCell()
    get_cell = {'rnn_relu': lambda cell_prefix: RNNCell(self._num_hidden, activation='relu', prefix=cell_prefix), 'rnn_tanh': lambda cell_prefix: RNNCell(self._num_hidden, activation='tanh', prefix=cell_prefix), 'lstm': lambda cell_prefix: LSTMCell(self._num_hidden, prefix=cell_prefix), 'gru': lambda cell_prefix: GRUCell(self._num_hidden, prefix=cell_prefix)}[self._mode]
    for i in range(self._num_layers):
        if self._bidirectional:
            stack.add(BidirectionalCell(get_cell('%sl%d_' % (self._prefix, i)), get_cell('%sr%d_' % (self._prefix, i)), output_prefix='%sbi_l%d_' % (self._prefix, i)))
        else:
            stack.add(get_cell('%sl%d_' % (self._prefix, i)))
        if self._dropout > 0 and i != self._num_layers - 1:
            stack.add(DropoutCell(self._dropout, prefix='%s_dropout%d_' % (self._prefix, i)))
    return stack