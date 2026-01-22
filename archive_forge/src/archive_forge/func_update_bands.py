from ipywidgets import Layout
from traitlets import List, Enum, Int, Bool
from traittypes import DataFrame
from bqplot import Figure, LinearScale, Lines, Label
from bqplot.marks import CATEGORY10
import numpy as np
def update_bands(self, *args):
    band_data = np.linspace(self.data_range[0], self.data_range[1], self.num_bands + 1)
    self.scaled_band_data = ((band_data - self.data_range[0]) / (self.data_range[1] - self.data_range[0]))[:, np.newaxis]
    n = len(self.data.index)
    if self.band_type == 'circle':
        t = np.linspace(0, 2 * np.pi, 1000)
        band_data_x, band_data_y = (self.scaled_band_data * np.cos(t), self.scaled_band_data * np.sin(t))
    elif self.band_type == 'polygon':
        t = np.linspace(0, 2 * np.pi, n + 1)
        band_data_x, band_data_y = (self.scaled_band_data * np.sin(t), self.scaled_band_data * np.cos(t))
    with self.bands.hold_sync():
        self.bands.x = band_data_x
        self.bands.y = band_data_y
    with self.band_labels.hold_sync():
        self.band_labels.x = self.scaled_band_data[:, 0]
        self.band_labels.y = [0.0] * (self.num_bands + 1)
        self.band_labels.text = ['{:.0%}'.format(b) for b in band_data]