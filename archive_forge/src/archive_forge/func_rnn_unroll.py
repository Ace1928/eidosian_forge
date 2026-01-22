import warnings
from ..model import save_checkpoint, load_checkpoint
from .rnn_cell import BaseRNNCell
def rnn_unroll(cell, length, inputs=None, begin_state=None, input_prefix='', layout='NTC'):
    """Deprecated. Please use cell.unroll instead"""
    warnings.warn('rnn_unroll is deprecated. Please call cell.unroll directly.')
    return cell.unroll(length=length, inputs=inputs, begin_state=begin_state, input_prefix=input_prefix, layout=layout)