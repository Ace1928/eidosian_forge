from kivy.logger import Logger
from kivy.uix.layout import Layout
from kivy.properties import NumericProperty, BooleanProperty, DictProperty, \
from math import ceil
from itertools import accumulate, product, chain, islice
from operator import sub
Orientation of the layout.

    :attr:`orientation` is an :class:`~kivy.properties.OptionProperty` and
    defaults to 'lr-tb'.

    Valid orientations are 'lr-tb', 'tb-lr', 'rl-tb', 'tb-rl', 'lr-bt',
    'bt-lr', 'rl-bt' and 'bt-rl'.

    .. versionadded:: 2.0.0

    .. note::

        'lr' means Left to Right.
        'rl' means Right to Left.
        'tb' means Top to Bottom.
        'bt' means Bottom to Top.
    