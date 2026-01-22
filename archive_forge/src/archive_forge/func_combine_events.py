from __future__ import absolute_import, division, print_function
import argparse
import contextlib
import numpy as np
def combine_events(events, delta, combine='mean'):
    """
    Combine all events within a certain range.

    Parameters
    ----------
    events : list or numpy array
        Events to be combined.
    delta : float
        Combination delta. All events within this `delta` are combined.
    combine : {'mean', 'left', 'right'}
        How to combine two adjacent events:

            - 'mean': replace by the mean of the two events
            - 'left': replace by the left of the two events
            - 'right': replace by the right of the two events

    Returns
    -------
    numpy array
        Combined events.

    """
    delta += 1e-12
    if len(events) <= 1:
        return events
    events = np.array(events, dtype=np.float)
    if events.ndim > 1:
        raise ValueError('only 1-dimensional events supported.')
    idx = 0
    left = events[idx]
    for right in events[1:]:
        if right - left <= delta:
            if combine == 'mean':
                left = events[idx] = 0.5 * (right + left)
            elif combine == 'left':
                left = events[idx] = left
            elif combine == 'right':
                left = events[idx] = right
            else:
                raise ValueError("don't know how to combine two events with %s" % combine)
        else:
            idx += 1
            left = events[idx] = right
    return events[:idx + 1]