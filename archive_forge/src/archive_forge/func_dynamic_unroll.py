from ...rnn import BidirectionalCell, SequentialRNNCell, ModifierCell, HybridRecurrentCell
from ...rnn.rnn_cell import _format_sequence, _get_begin_state, _mask_sequence_variable_length
from ... import tensor_types
from ....base import _as_list
def dynamic_unroll(cell, inputs, begin_state, drop_inputs=0, drop_outputs=0, layout='TNC', valid_length=None):
    """Unrolls an RNN cell across time steps.

    Currently, 'TNC' is a preferred layout. unroll on the input of this layout
    runs much faster.

    Parameters
    ----------
    cell : an object whose base class is RNNCell.
        The RNN cell to run on the input sequence.
    inputs : Symbol
        It should have shape (batch_size, length, ...) if `layout` is 'NTC',
        or (length, batch_size, ...) if `layout` is 'TNC'.
    begin_state : nested list of Symbol
        The initial states of the RNN sequence.
    drop_inputs : float, default 0.
        The dropout rate for inputs. Won't apply dropout if it equals 0.
    drop_outputs : float, default 0.
        The dropout rate for outputs. Won't apply dropout if it equals 0.
    layout : str, optional
        `layout` of input symbol. Only used if inputs
        is a single Symbol.
    valid_length : Symbol, NDArray or None
        `valid_length` specifies the length of the sequences in the batch without padding.
        This option is especially useful for building sequence-to-sequence models where
        the input and output sequences would potentially be padded.
        If `valid_length` is None, all sequences are assumed to have the same length.
        If `valid_length` is a Symbol or NDArray, it should have shape (batch_size,).
        The ith element will be the length of the ith sequence in the batch.
        The last valid state will be return and the padded outputs will be masked with 0.
        Note that `valid_length` must be smaller or equal to `length`.

    Returns
    -------
    outputs : Symbol
        the output of the RNN from this unrolling.

    states : list of Symbol
        The new state of this RNN after this unrolling.
        The type of this symbol is same as the output of `begin_state`.

    Examples
    --------
    >>> seq_len = 3
    >>> batch_size = 2
    >>> input_size = 5
    >>> cell = mx.gluon.rnn.LSTMCell(input_size, prefix='rnn_')
    >>> cell.initialize(ctx=mx.cpu())
    >>> rnn_data = mx.nd.normal(loc=0, scale=1, shape=(seq_len, batch_size, input_size))
    >>> state_shape = (batch_size, input_size)
    >>> states = [mx.nd.normal(loc=0, scale=1, shape=state_shape) for i in range(2)]
    >>> valid_length = mx.nd.array([2, 3])
    >>> output, states = mx.gluon.contrib.rnn.rnn_cell.dynamic_unroll(cell, rnn_data, states,
                                                                      valid_length=valid_length,
                                                                      layout='TNC')
    >>> print(output)
    [[[ 0.00767238  0.00023103  0.03973929 -0.00925503 -0.05660512]
      [ 0.00881535  0.05428379 -0.02493718 -0.01834097  0.02189514]]
     [[-0.00676967  0.01447039  0.01287002 -0.00574152 -0.05734247]
      [ 0.01568508  0.02650866 -0.04270559 -0.04328435  0.00904011]]
     [[ 0.          0.          0.          0.          0.        ]
      [ 0.01055336  0.02734251 -0.03153727 -0.03742751 -0.01378113]]]
     <NDArray 3x2x5 @cpu(0)>
    """
    inputs, axis, F, _ = _format_sequence(0, inputs, layout, True)
    if axis != 0:
        axes = list(range(len(layout)))
        tmp = axes[0]
        axes[0] = axes[axis]
        axes[axis] = tmp
        inputs = F.transpose(inputs, axes=axes)
    states = begin_state
    if drop_inputs:
        inputs = F.Dropout(inputs, p=drop_inputs, axes=(axis,))
    if valid_length is None:

        def loop_body(inputs, states):
            return cell(inputs, states)
    else:
        zeros = []
        for s in states:
            zeros.append(F.zeros_like(s))
        states = list(_as_list(states))
        states.append(F.zeros(1))

        def loop_body(inputs, states):
            cell_states = states[:-1]
            iter_no = states[-1]
            out, new_states = cell(inputs, cell_states)
            for i, state in enumerate(cell_states):
                new_states[i] = F.where(F.broadcast_greater(valid_length, iter_no), new_states[i], state)
            new_states.append(iter_no + 1)
            return (out, new_states)
    outputs, states = F.contrib.foreach(loop_body, inputs, states)
    if drop_outputs:
        outputs = F.Dropout(outputs, p=drop_outputs, axes=(axis,))
    if valid_length is not None:
        if axis != 0:
            outputs = F.transpose(outputs, axes)
        outputs = F.SequenceMask(outputs, sequence_length=valid_length, use_sequence_length=True, axis=axis)
        return (outputs, states[:-1])
    else:
        if axis != 0:
            outputs = F.transpose(outputs, axes)
        return (outputs, states)