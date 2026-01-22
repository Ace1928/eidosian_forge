import warnings
from ..model import save_checkpoint, load_checkpoint
from .rnn_cell import BaseRNNCell
def save_rnn_checkpoint(cells, prefix, epoch, symbol, arg_params, aux_params):
    """Save checkpoint for model using RNN cells.
    Unpacks weight before saving.

    Parameters
    ----------
    cells : mxnet.rnn.RNNCell or list of RNNCells
        The RNN cells used by this symbol.
    prefix : str
        Prefix of model name.
    epoch : int
        The epoch number of the model.
    symbol : Symbol
        The input symbol
    arg_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's weights.
    aux_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's auxiliary states.

    Notes
    -----
    - ``prefix-symbol.json`` will be saved for symbol.
    - ``prefix-epoch.params`` will be saved for parameters.
    """
    if isinstance(cells, BaseRNNCell):
        cells = [cells]
    for cell in cells:
        arg_params = cell.unpack_weights(arg_params)
    save_checkpoint(prefix, epoch, symbol, arg_params, aux_params)