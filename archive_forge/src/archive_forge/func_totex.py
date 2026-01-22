from __future__ import absolute_import, division, print_function
import numpy as np
from . import chords, beats, notes, onsets, tempo
from .beats import BeatEvaluation, BeatMeanEvaluation
from .chords import ChordEvaluation, ChordMeanEvaluation, ChordSumEvaluation
from .key import KeyEvaluation, KeyMeanEvaluation
from .notes import NoteEvaluation, NoteMeanEvaluation, NoteSumEvaluation
from .onsets import OnsetEvaluation, OnsetMeanEvaluation, OnsetSumEvaluation
from .tempo import TempoEvaluation, TempoMeanEvaluation
def totex(eval_objects, metric_names=None, float_format='{:.3f}', **kwargs):
    """
    Format the given evaluation objects as a LaTeX table.

    Parameters
    ----------
    eval_objects : list
        Evaluation objects.
    metric_names : list of tuples, optional
        List of tuples defining the name of the property corresponding to the
        metric, and the metric label e.g. ('fp', 'False Positives').
    float_format : str, optional
        How to format the metrics.

    Returns
    -------
    str
        LaTeX table representation of the evaluation objects.

    Notes
    -----
    If no `metric_names` are given, they will be extracted from the first
    evaluation object.

    """
    if metric_names is None:
        metric_names = eval_objects[0].METRIC_NAMES
    metric_names, metric_labels = list(zip(*metric_names))
    lines = ['Name & ' + ' & '.join(metric_labels) + '\\\\']
    for e in eval_objects:
        values = [float_format.format(getattr(e, mn)) for mn in metric_names]
        lines.append(e.name + ' & ' + ' & '.join(values) + '\\\\')
    return '\n'.join(lines)