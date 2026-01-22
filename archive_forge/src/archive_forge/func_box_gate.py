from collections.abc import Iterable
import warnings
from typing import Sequence
def box_gate(self, layer, wires, text='', box_options=None, text_options=None, **kwargs):
    """Draws a box and adds label text to its center.

        Args:
            layer (int): x coordinate for the box center
            wires (Union[int, Iterable[int]]): y locations to include inside the box. Only min and max
                of an Iterable affect the output
            text (str): string to print at the box's center

        Keyword Args:
            box_options=None (dict): any matplotlib keywords for the ``plt.Rectangle`` patch
            text_options=None (dict): any matplotlib keywords for the text
            extra_width (float): extra box width
            autosize (bool): whether to rotate and shrink text to fit within the box
            active_wire_notches (bool): whether or not to add notches indicating active wires.
                Defaults to ``True``.

        **Example**

        .. code-block:: python

            drawer = MPLDrawer(n_wires=2, n_layers=1)

            drawer.box_gate(layer=0, wires=(0, 1), text="CY")

        .. figure:: ../../_static/drawer/box_gates.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

        .. details::
            :title: Usage Details

        This method can accept two different sets of design keywords. ``box_options`` takes
        `Rectangle keywords <https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html>`_
        , and ``text_options`` accepts
        `Matplotlib Text keywords <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html>`_ .

        .. code-block:: python

            box_options = {'facecolor': 'lightcoral', 'edgecolor': 'maroon', 'linewidth': 5}
            text_options = {'fontsize': 'xx-large', 'color': 'maroon'}

            drawer = MPLDrawer(n_wires=2, n_layers=1)

            drawer.box_gate(layer=0, wires=(0, 1), text="CY",
                box_options=box_options, text_options=text_options)

        .. figure:: ../../_static/drawer/box_gates_formatted.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

        By default, text is rotated and/or shrunk to fit within the box. This behavior can be turned off
        with the ``autosize=False`` keyword.

        .. code-block:: python

            drawer = MPLDrawer(n_layers=4, n_wires=2)

            drawer.box_gate(layer=0, wires=0, text="A longer label")
            drawer.box_gate(layer=0, wires=1, text="Label")

            drawer.box_gate(layer=1, wires=(0,1), text="long multigate label")

            drawer.box_gate(layer=3, wires=(0,1), text="Not autosized label", autosize=False)

        .. figure:: ../../_static/drawer/box_gates_autosized.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

        """
    extra_width = kwargs.get('extra_width', 0)
    autosize = kwargs.get('autosize', True)
    active_wire_notches = kwargs.get('active_wire_notches', True)
    if box_options is None:
        box_options = {}
    if 'zorder' not in box_options:
        box_options['zorder'] = 2
    new_text_options = {'zorder': 3, 'ha': 'center', 'va': 'center', 'fontsize': self.fontsize}
    if text_options is not None:
        new_text_options.update(text_options)
    wires = _to_tuple(wires)
    box_min = min(wires)
    box_max = max(wires)
    box_center = (box_max + box_min) / 2.0
    x_loc = layer - self._box_length / 2.0 - extra_width / 2.0 + self._pad
    y_loc = box_min - self._box_length / 2.0 + self._pad
    box_height = box_max - box_min + self._box_length - 2 * self._pad
    box_width = self._box_length + extra_width - 2 * self._pad
    box = patches.FancyBboxPatch((x_loc, y_loc), box_width, box_height, boxstyle=self._boxstyle, **box_options)
    self._ax.add_patch(box)
    text_obj = self._ax.text(layer, box_center, text, **new_text_options)
    if active_wire_notches and len(wires) != box_max - box_min + 1:
        notch_options = box_options.copy()
        notch_options['zorder'] += -1
        for wire in wires:
            self._add_notch(layer, wire, extra_width, notch_options)
    if autosize:
        margin = 0.1
        max_width = box_width - margin + 2 * self._pad
        max_height = box_height - 2 * margin + 2 * self._pad
        w, h = self._text_dims(text_obj)
        if box_min != box_max and w > max_width and (w > h):
            text_obj.set_rotation(90)
            w, h = self._text_dims(text_obj)
        current_fontsize = text_obj.get_fontsize()
        for s in range(int(current_fontsize), 1, -1):
            if w < max_width and h < max_height:
                break
            text_obj.set_fontsize(s)
            w, h = self._text_dims(text_obj)