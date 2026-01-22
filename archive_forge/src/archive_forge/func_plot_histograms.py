from collections import abc, defaultdict
import datetime
from itertools import cycle
from typing import Any, cast, Dict, Iterator, List, Optional, Tuple, Union, Sequence
import matplotlib as mpl
import matplotlib.pyplot as plt
import google.protobuf.json_format as json_format
import cirq
from cirq_google.api import v2
def plot_histograms(self, keys: Sequence[str], ax: Optional[plt.Axes]=None, *, labels: Optional[Sequence[str]]=None) -> plt.Axes:
    """Plots integrated histograms of metric values corresponding to keys

        Args:
            keys: List of metric keys for which an integrated histogram should be plot
            ax: The axis to plot on. If None, we generate one.
            labels: Optional label that will be used in the legend.

        Returns:
            The axis that was plotted on.

        Raises:
            ValueError: If the metric values are not single floats.
        """
    show_plot = not ax
    if not ax:
        fig, ax = plt.subplots(1, 1)
        ax = cast(plt.Axes, ax)
    if isinstance(keys, str):
        keys = [keys]
    if not labels:
        labels = keys
    colors = ['b', 'r', 'k', 'g', 'c', 'm']
    for key, label, color in zip(keys, labels, cycle(colors)):
        metrics = self[key]
        if not all((len(k) == 1 for k in metrics.values())):
            raise ValueError('Histograms are only supported if all values in a metric ' + 'are single metric values.' + f'{key} has metric values {metrics.values()}')
        cirq.integrated_histogram([self.value_to_float(v) for v in metrics.values()], ax, label=label, color=color, title=key.replace('_', ' ').title())
    if show_plot:
        fig.show()
    return ax