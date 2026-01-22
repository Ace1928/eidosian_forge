import os
from math import log10
from time import perf_counter
import numpy as np
from .. import debug as debug
from .. import functions as fn
from .. import getConfigOption
from ..graphicsItems.GradientEditorItem import addGradientListToDocstring
from ..graphicsItems.ImageItem import ImageItem
from ..graphicsItems.InfiniteLine import InfiniteLine
from ..graphicsItems.LinearRegionItem import LinearRegionItem
from ..graphicsItems.ROI import ROI
from ..graphicsItems.ViewBox import ViewBox
from ..graphicsItems.VTickGroup import VTickGroup
from ..Qt import QtCore, QtGui, QtWidgets
from ..SignalProxy import SignalProxy
from . import ImageViewTemplate_generic as ui_template
def setImage(self, img, autoRange=True, autoLevels=True, levels=None, axes=None, xvals=None, pos=None, scale=None, transform=None, autoHistogramRange=True, levelMode=None):
    """
        Set the image to be displayed in the widget.

        Parameters
        ----------
        img : np.ndarray
            The image to be displayed. See :func:`ImageItem.setImage` and *notes* below.
        autoRange : bool
            Whether to scale/pan the view to fit the image.
        autoLevels : bool
            Whether to update the white/black levels to fit the image.
        levels : tuple
            (min, max) white and black level values to use.
        axes : dict
            Dictionary indicating the interpretation for each axis. This is only needed to override the default guess.
            Format is::

                {'t':0, 'x':1, 'y':2, 'c':3};
        xvals : np.ndarray
            1D array of values corresponding to the first axis in a 3D image. For video, this array should contain
            the time of each frame.
        pos
            Change the position of the displayed image
        scale
            Change the scale of the displayed image
        transform
            Set the transform of the displayed image. This option overrides *pos* and *scale*.
        autoHistogramRange : bool
            If True, the histogram y-range is automatically scaled to fit the image data.
        levelMode : str
            If specified, this sets the user interaction mode for setting image levels. Options are 'mono',
            which provides a single level control for all image channels, and 'rgb' or 'rgba', which provide
            individual controls for each channel.

        Notes
        -----
        For backward compatibility, image data is assumed to be in column-major order (column, row).
        However, most image data is stored in row-major order (row, column) and will need to be
        transposed before calling setImage()::
        
            imageview.setImage(imagedata.T)
            
        This requirement can be changed by the ``imageAxisOrder``
        :ref:`global configuration option <apiref_config>`.
        """
    profiler = debug.Profiler()
    if hasattr(img, 'implements') and img.implements('MetaArray'):
        img = img.asarray()
    if not isinstance(img, np.ndarray):
        required = ['dtype', 'max', 'min', 'ndim', 'shape', 'size']
        if not all((hasattr(img, attr) for attr in required)):
            raise TypeError('Image must be NumPy array or any object that provides compatible attributes/methods:\n  %s' % str(required))
    self.image = img
    self.imageDisp = None
    if levelMode is not None:
        self.ui.histogram.setLevelMode(levelMode)
    profiler()
    if axes is None:
        x, y = (0, 1) if self.imageItem.axisOrder == 'col-major' else (1, 0)
        if img.ndim == 2:
            self.axes = {'t': None, 'x': x, 'y': y, 'c': None}
        elif img.ndim == 3:
            if img.shape[2] <= 4:
                self.axes = {'t': None, 'x': x, 'y': y, 'c': 2}
            else:
                self.axes = {'t': 0, 'x': x + 1, 'y': y + 1, 'c': None}
        elif img.ndim == 4:
            self.axes = {'t': 0, 'x': x + 1, 'y': y + 1, 'c': 3}
        else:
            raise Exception('Can not interpret image with dimensions %s' % str(img.shape))
    elif isinstance(axes, dict):
        self.axes = axes.copy()
    elif isinstance(axes, list) or isinstance(axes, tuple):
        self.axes = {}
        for i in range(len(axes)):
            self.axes[axes[i]] = i
    else:
        raise Exception("Can not interpret axis specification %s. Must be like {'t': 2, 'x': 0, 'y': 1} or ('t', 'x', 'y', 'c')" % str(axes))
    for x in ['t', 'x', 'y', 'c']:
        self.axes[x] = self.axes.get(x, None)
    axes = self.axes
    if xvals is not None:
        self.tVals = xvals
    elif axes['t'] is not None:
        if hasattr(img, 'xvals'):
            try:
                self.tVals = img.xvals(axes['t'])
            except:
                self.tVals = np.arange(img.shape[axes['t']])
        else:
            self.tVals = np.arange(img.shape[axes['t']])
    profiler()
    self.currentIndex = 0
    self.updateImage(autoHistogramRange=autoHistogramRange)
    if levels is None and autoLevels:
        self.autoLevels()
    if levels is not None:
        self.setLevels(*levels)
    if self.ui.roiBtn.isChecked():
        self.roiChanged()
    profiler()
    if self.axes['t'] is not None:
        self.ui.roiPlot.setXRange(self.tVals.min(), self.tVals.max())
        self.frameTicks.setXVals(self.tVals)
        self.timeLine.setValue(0)
        if len(self.tVals) > 1:
            start = self.tVals.min()
            stop = self.tVals.max() + abs(self.tVals[-1] - self.tVals[0]) * 0.02
        elif len(self.tVals) == 1:
            start = self.tVals[0] - 0.5
            stop = self.tVals[0] + 0.5
        else:
            start = 0
            stop = 1
        for s in [self.timeLine, self.normRgn]:
            s.setBounds([start, stop])
    profiler()
    if transform is None:
        transform = QtGui.QTransform()
        if pos is not None:
            transform.translate(*pos)
        if scale is not None:
            transform.scale(*scale)
    self.imageItem.setTransform(transform)
    profiler()
    if autoRange:
        self.autoRange()
    self.roiClicked()
    profiler()