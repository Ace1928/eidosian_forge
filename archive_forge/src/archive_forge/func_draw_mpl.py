from functools import wraps
from importlib.metadata import distribution
import warnings
import pennylane as qml
from .tape_mpl import tape_mpl
from .tape_text import tape_text
def draw_mpl(qnode, wire_order=None, show_all_wires=False, decimals=None, expansion_strategy=None, style=None, *, fig=None, **kwargs):
    """Draw a qnode with matplotlib

    Args:
        qnode (.QNode or Callable): the input QNode/quantum function that is to be drawn.

    Keyword Args:
        wire_order (Sequence[Any]): the order (from top to bottom) to print the wires of the circuit
        show_all_wires (bool): If True, all wires, including empty wires, are printed.
        decimals (int): How many decimal points to include when formatting operation parameters.
            Default ``None`` will omit parameters from operation labels.
        style (str): visual style of plot. Valid strings are ``{'black_white', 'black_white_dark', 'sketch',
            'pennylane', 'pennylane_sketch', 'sketch_dark', 'solarized_light', 'solarized_dark', 'default'}``.
            If no style is specified, the global style set with :func:`~.use_style` will be used, and the
            initial default is 'black_white'. If you would like to use your environment's current rcParams,
            set ``style`` to "rcParams". Setting style does not modify matplotlib global plotting settings.
        fontsize (float or str): fontsize for text. Valid strings are
            ``{'xx-small', 'x-small', 'small', 'medium', large', 'x-large', 'xx-large'}``.
            Default is ``14``.
        wire_options (dict): matplotlib formatting options for the wire lines
        label_options (dict): matplotlib formatting options for the wire labels
        active_wire_notches (bool): whether or not to add notches indicating active wires.
            Defaults to ``True``.
        expansion_strategy (str): The strategy to use when circuit expansions or decompositions
            are required.

            - ``gradient``: The QNode will attempt to decompose
              the internal circuit such that all circuit operations are supported by the gradient
              method.

            - ``device``: The QNode will attempt to decompose the internal circuit
              such that all circuit operations are natively supported by the device.
        fig (None or matplotlib.Figure): Matplotlib figure to plot onto. If None, then create a new figure

    Returns:
        A function that has the same argument signature as ``qnode``. When called,
        the function will draw the QNode as a tuple of (``matplotlib.figure.Figure``,
        ``matplotlib.axes._axes.Axes``)

    **Example**:

    .. code-block:: python

        dev = qml.device('lightning.qubit', wires=(0,1,2,3))

        @qml.qnode(dev)
        def circuit(x, z):
            qml.QFT(wires=(0,1,2,3))
            qml.IsingXX(1.234, wires=(0,2))
            qml.Toffoli(wires=(0,1,2))
            qml.CSWAP(wires=(0,2,3))
            qml.RX(x, wires=0)
            qml.CRZ(z, wires=(3,0))
            return qml.expval(qml.Z(0))


        fig, ax = qml.draw_mpl(circuit)(1.2345,1.2345)
        fig.show()

    .. figure:: ../../_static/draw_mpl/main_example.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    .. details::
        :title: Usage Details

        **Decimals:**

        The keyword ``decimals`` controls how many decimal points to include when labelling the operations.
        The default value ``None`` omits parameters for brevity.

        .. code-block:: python

            @qml.qnode(dev)
            def circuit2(x, y):
                qml.RX(x, wires=0)
                qml.Rot(*y, wires=0)
                return qml.expval(qml.Z(0))

            fig, ax = qml.draw_mpl(circuit2, decimals=2)(1.23456, [1.2345,2.3456,3.456])
            fig.show()

        .. figure:: ../../_static/draw_mpl/decimals.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

        **Wires:**

        The keywords ``wire_order`` and ``show_all_wires`` control the location of wires from top to bottom.

        .. code-block:: python

            fig, ax = qml.draw_mpl(circuit, wire_order=[3,2,1,0])(1.2345,1.2345)
            fig.show()

        .. figure:: ../../_static/draw_mpl/wire_order.png
                :align: center
                :width: 60%
                :target: javascript:void(0);

        If a wire is in ``wire_order``, but not in the ``tape``, it will be omitted by default.  Only by selecting
        ``show_all_wires=True`` will empty wires be displayed.

        .. code-block:: python

            fig, ax = qml.draw_mpl(circuit, wire_order=["aux"], show_all_wires=True)(1.2345,1.2345)
            fig.show()

        .. figure:: ../../_static/draw_mpl/show_all_wires.png
                :align: center
                :width: 60%
                :target: javascript:void(0);

        **Integration with matplotlib:**

        This function returns matplotlib figure and axes objects. Using these objects,
        users can perform further customization of the graphic.

        .. code-block:: python

            fig, ax = qml.draw_mpl(circuit)(1.2345,1.2345)
            fig.suptitle("My Circuit", fontsize="xx-large")

            options = {'facecolor': "white", 'edgecolor': "#f57e7e", "linewidth": 6, "zorder": -1}
            box1 = plt.Rectangle((-0.5, -0.5), width=3.0, height=4.0, **options)
            ax.add_patch(box1)

            ax.annotate("CSWAP", xy=(3, 2.5), xycoords='data', xytext=(3.8,1.5), textcoords='data',
                        arrowprops={'facecolor': 'black'}, fontsize=14)
            fig.show()

        .. figure:: ../../_static/draw_mpl/postprocessing.png
                :align: center
                :width: 60%
                :target: javascript:void(0);

        **Formatting:**

        PennyLane has inbuilt styles for controlling the appearance of the circuit drawings.
        All available styles can be determined by evaluating ``qml.drawer.available_styles()``.
        Any available string can then be passed via the kwarg ``style`` to change the settings for
        that plot. This will not affect style settings for subsequent matplotlib plots.

        .. code-block:: python

            fig, ax = qml.draw_mpl(circuit, style='sketch')(1.2345,1.2345)
            fig.show()


        .. figure:: ../../_static/draw_mpl/sketch_style.png
                :align: center
                :width: 60%
                :target: javascript:void(0);

        You can also control the appearance with matplotlib's provided tools, see the
        `matplotlib docs <https://matplotlib.org/stable/tutorials/introductory/customizing.html>`_ .
        For example, we can customize ``plt.rcParams``. To use a customized appearance based on matplotlib's
        ``plt.rcParams``, ``qml.draw_mpl`` must be run with ``style="rcParams"``:

        .. code-block:: python

            plt.rcParams['patch.facecolor'] = 'mistyrose'
            plt.rcParams['patch.edgecolor'] = 'maroon'
            plt.rcParams['text.color'] = 'maroon'
            plt.rcParams['font.weight'] = 'bold'
            plt.rcParams['patch.linewidth'] = 4
            plt.rcParams['patch.force_edgecolor'] = True
            plt.rcParams['lines.color'] = 'indigo'
            plt.rcParams['lines.linewidth'] = 5
            plt.rcParams['figure.facecolor'] = 'ghostwhite'

            fig, ax = qml.draw_mpl(circuit, style="rcParams")(1.2345,1.2345)
            fig.show()

        .. figure:: ../../_static/draw_mpl/rcparams.png
                :align: center
                :width: 60%
                :target: javascript:void(0);

        The wires and wire labels can be manually formatted by passing in dictionaries of
        keyword-value pairs of matplotlib options. ``wire_options`` accepts options for lines,
        and ``label_options`` accepts text options.

        .. code-block:: python

            fig, ax = qml.draw_mpl(circuit, wire_options={'color':'teal', 'linewidth': 5},
                        label_options={'size': 20})(1.2345,1.2345)
            fig.show()

        .. figure:: ../../_static/draw_mpl/wires_labels.png
                :align: center
                :width: 60%
                :target: javascript:void(0);

    """
    if catalyst_qjit(qnode):
        qnode = qnode.user_function
    if hasattr(qnode, 'construct'):
        return _draw_mpl_qnode(qnode, wire_order=wire_order, show_all_wires=show_all_wires, decimals=decimals, expansion_strategy=expansion_strategy, style=style, fig=fig, **kwargs)
    if expansion_strategy is not None:
        warnings.warn('When the input to qml.draw is not a QNode, the expansion_strategy argument is ignored.', UserWarning)

    @wraps(qnode)
    def wrapper(*args, **kwargs):
        tape = qml.tape.make_qscript(qnode)(*args, **kwargs)
        _wire_order = wire_order or tape.wires
        return tape_mpl(tape, wire_order=_wire_order, show_all_wires=show_all_wires, decimals=decimals, style=style, fig=fig, **kwargs)
    return wrapper