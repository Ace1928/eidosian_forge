import torch
from typing import Any, Set, Dict, List, Tuple, OrderedDict
from collections import OrderedDict as OrdDict
def generate_plot_visualization(self, feature_filter: str, module_fqn_filter: str=''):
    """
        Takes in a feature and optional module_filter and plots of the desired data.

        For per channel features, it averages the value across the channels and plots a point
        per module. The reason for this is that for models with hundreds of channels, it can
        be hard to differentiate one channel line from another, and so the point of generating
        a single average point per module is to give a sense of general trends that encourage
        further deep dives.

        Note:
            Only features in the report that have tensor value data are plottable by this class
            When the tensor information is plotted, it will plot:
                idx as the x val, feature value as the y_val
            When the channel information is plotted, it will plot:
                the first idx of each module as the x val, feature value as the y_val [for each channel]
                The reason for this is that we want to be able to compare values across the
                channels for same layer, and it will be hard if values are staggered by idx
                This means each module is represented by only 1 x value
        Args:
            feature_filter (str): Filters the features presented to only those that
                contain this filter substring
            module_fqn_filter (str, optional): Only includes modules that contains this string
                Default = "", results in all the modules in the reports to be visible in the table

        Example Use:
            >>> # xdoctest: +SKIP("undefined variables")
            >>> mod_report_visualizer.generate_plot_visualization(
            ...     feature_filter = "per_channel_min",
            ...     module_fqn_filter = "block1"
            ... )
            >>> # outputs line plot of per_channel_min information for all
            >>> # modules in block1 of model each channel gets it's own line,
            >>> # and it's plotted across the in-order modules on the x-axis
        """
    if not got_matplotlib:
        print('make sure to install matplotlib and try again.')
        return None
    x_data, y_data, data_per_channel = self._get_plottable_data(feature_filter, module_fqn_filter)
    ax = plt.subplot()
    ax.set_ylabel(feature_filter)
    ax.set_title(feature_filter + ' Plot')
    plt.xticks(x_data)
    if data_per_channel:
        ax.set_xlabel('First idx of module')
        num_modules = len(y_data[0])
        num_channels = len(y_data)
        avg_vals = [sum(y_data[:][index]) / num_channels for index in range(num_modules)]
        ax.plot(x_data, avg_vals, label=f'Average Value Across {num_channels} Channels')
        ax.legend(loc='upper right')
    else:
        ax.set_xlabel('idx')
        ax.plot(x_data, y_data)
    plt.show()