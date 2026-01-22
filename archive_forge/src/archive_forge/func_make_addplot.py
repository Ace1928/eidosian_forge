import matplotlib.dates  as mdates
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.axes   as mpl_axes
import matplotlib.figure as mpl_fig
import pandas as pd
import numpy  as np
import copy
import io
import os
import math
import warnings
import statistics as stat
from itertools import cycle
from mplfinance._utils import _construct_aline_collections
from mplfinance._utils import _construct_hline_collections
from mplfinance._utils import _construct_vline_collections
from mplfinance._utils import _construct_tline_collections
from mplfinance._utils import _construct_mpf_collections
from mplfinance._utils import _construct_pnf_scatter
from mplfinance._widths import _determine_width_config
from mplfinance._utils import _updown_colors
from mplfinance._utils import IntegerIndexDateTimeFormatter
from mplfinance._utils import _mscatter
from mplfinance._utils import _check_and_convert_xlim_configuration
from mplfinance import _styles
from mplfinance._arg_validators import _check_and_prepare_data, _mav_validator, _label_validator
from mplfinance._arg_validators import _get_valid_plot_types, _fill_between_validator
from mplfinance._arg_validators import _process_kwargs, _validate_vkwargs_dict
from mplfinance._arg_validators import _kwarg_not_implemented, _bypass_kwarg_validation
from mplfinance._arg_validators import _hlines_validator, _vlines_validator
from mplfinance._arg_validators import _alines_validator, _tlines_validator
from mplfinance._arg_validators import _scale_padding_validator, _yscale_validator
from mplfinance._arg_validators import _valid_panel_id, _check_for_external_axes
from mplfinance._arg_validators import _xlim_validator, _mco_validator, _is_marketcolor_object
from mplfinance._panels import _build_panels
from mplfinance._panels import _set_ticks_on_bottom_panel_only
from mplfinance._helpers import _determine_format_string
from mplfinance._helpers import _list_of_dict
from mplfinance._helpers import _num_or_seq_of_num
from mplfinance._helpers import _adjust_color_brightness
def make_addplot(data, **kwargs):
    """
    Take data (pd.Series, pd.DataFrame, np.ndarray of floats, list of floats), and
    kwargs (see valid_addplot_kwargs_table) and construct a correctly structured dict
    to be passed into plot() using kwarg `addplot`.  
    NOTE WELL: len(data) here must match the len(data) passed into plot()
    """
    if not isinstance(data, (pd.Series, pd.DataFrame, np.ndarray, list)):
        raise TypeError('Wrong type for data, in make_addplot()')
    config = _process_kwargs(kwargs, _valid_addplot_kwargs())
    if config['scatter'] == True and config['type'] == 'line':
        config['type'] = 'scatter'
    return dict(data=data, **config)