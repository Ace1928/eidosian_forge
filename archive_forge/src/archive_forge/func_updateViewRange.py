import math
import sys
import weakref
from copy import deepcopy
import numpy as np
from ... import debug as debug
from ... import functions as fn
from ... import getConfigOption
from ...Point import Point
from ...Qt import QtCore, QtGui, QtWidgets, isQObjectAlive, QT_LIB
from ..GraphicsWidget import GraphicsWidget
from ..ItemGroup import ItemGroup
from .ViewBoxMenu import ViewBoxMenu
def updateViewRange(self, forceX=False, forceY=False):
    viewRange = [self.state['targetRange'][0][:], self.state['targetRange'][1][:]]
    changed = [False, False]
    aspect = self.state['aspectLocked']
    tr = self.targetRect()
    bounds = self.rect()
    limits = self._effectiveLimits()
    minRng = [self.state['limits']['xRange'][0], self.state['limits']['yRange'][0]]
    maxRng = [self.state['limits']['xRange'][1], self.state['limits']['yRange'][1]]
    for axis in [0, 1]:
        if limits[axis][0] is None and limits[axis][1] is None and (minRng[axis] is None) and (maxRng[axis] is None):
            continue
        if limits[axis][0] is not None and limits[axis][1] is not None:
            if maxRng[axis] is not None:
                maxRng[axis] = min(maxRng[axis], limits[axis][1] - limits[axis][0])
            else:
                maxRng[axis] = limits[axis][1] - limits[axis][0]
    if aspect is not False and 0 not in [aspect, tr.height(), bounds.height(), bounds.width()]:
        targetRatio = tr.width() / tr.height() if tr.height() != 0 else 1
        viewRatio = (bounds.width() / bounds.height() if bounds.height() != 0 else 1) / aspect
        viewRatio = 1 if viewRatio == 0 else viewRatio
        dy = 0.5 * (tr.width() / viewRatio - tr.height())
        dx = 0.5 * (tr.height() * viewRatio - tr.width())
        rangeY = [self.state['targetRange'][1][0] - dy, self.state['targetRange'][1][1] + dy]
        rangeX = [self.state['targetRange'][0][0] - dx, self.state['targetRange'][0][1] + dx]
        canidateRange = [rangeX, rangeY]
        if forceX:
            ax = 0
        elif forceY:
            ax = 1
        else:
            ax = 0 if targetRatio > viewRatio else 1
            target = 0 if ax == 1 else 1
            if maxRng is not None or minRng is not None:
                diff = canidateRange[target][1] - canidateRange[target][0]
                if maxRng[target] is not None and diff > maxRng[target] or (minRng[target] is not None and diff < minRng[target]):
                    viewRange[ax] = canidateRange[ax]
                    self.state['viewRange'][ax] = viewRange[ax]
                    self._resetTarget(force=True)
                    ax = target
        if ax == 0:
            if dy != 0:
                changed[1] = True
            viewRange[1] = rangeY
        else:
            if dx != 0:
                changed[0] = True
            viewRange[0] = rangeX
    for axis in [0, 1]:
        range = viewRange[axis][1] - viewRange[axis][0]
        if minRng[axis] is not None and minRng[axis] > range:
            viewRange[axis][1] = viewRange[axis][0] + minRng[axis]
            self.state['targetRange'][axis] = viewRange[axis]
        if maxRng[axis] is not None and maxRng[axis] < range:
            viewRange[axis][1] = viewRange[axis][0] + maxRng[axis]
            self.state['targetRange'][axis] = viewRange[axis]
        if limits[axis][0] is not None and viewRange[axis][0] < limits[axis][0]:
            delta = limits[axis][0] - viewRange[axis][0]
            viewRange[axis][0] += delta
            viewRange[axis][1] += delta
            self.state['targetRange'][axis] = viewRange[axis]
        if limits[axis][1] is not None and viewRange[axis][1] > limits[axis][1]:
            delta = viewRange[axis][1] - limits[axis][1]
            viewRange[axis][0] -= delta
            viewRange[axis][1] -= delta
            self.state['targetRange'][axis] = viewRange[axis]
    thresholds = [(viewRange[axis][1] - viewRange[axis][0]) * 1e-09 for axis in (0, 1)]
    changed = [abs(viewRange[axis][0] - self.state['viewRange'][axis][0]) > thresholds[axis] or abs(viewRange[axis][1] - self.state['viewRange'][axis][1]) > thresholds[axis] for axis in (0, 1)]
    self.state['viewRange'] = viewRange
    if any(changed):
        self._matrixNeedsUpdate = True
        self.update()
        for ax in [0, 1]:
            if not changed[ax]:
                continue
            link = self.linkedView(ax)
            if link is not None:
                link.linkedViewChanged(self, ax)
        if changed[0]:
            self.sigXRangeChanged.emit(self, tuple(self.state['viewRange'][0]))
        if changed[1]:
            self.sigYRangeChanged.emit(self, tuple(self.state['viewRange'][1]))
        self.sigRangeChanged.emit(self, self.state['viewRange'], changed)