def use_style(style: str):
    """Set a style setting. Reset to default style using ``use_style('black_white')``

    Args:
        style (str): A style specification.

    Current styles:

    * ``'default'``
    * ``'black_white'``
    * ``'black_white_dark'``
    * ``'sketch'``
    * ``'pennylane'``
    * ``'pennylane_sketch'``
    * ``'sketch_dark'``
    * ``'solarized_light'``
    * ``'solarized_dark'``

    **Example**:

    .. code-block:: python

        qml.drawer.use_style('black_white')

        @qml.qnode(qml.device('lightning.qubit', wires=(0,1,2,3)))
        def circuit(x, z):
            qml.QFT(wires=(0,1,2,3))
            qml.Toffoli(wires=(0,1,2))
            qml.CSWAP(wires=(0,2,3))
            qml.RX(x, wires=0)
            qml.CRZ(z, wires=(3,0))
            return qml.expval(qml.Z(0))


        fig, ax = qml.draw_mpl(circuit)(1.2345,1.2345)
        fig.show()

    .. figure:: ../../_static/style/black_white_style.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    """
    global __current_style_fn
    if style in _styles_map:
        __current_style_fn = _styles_map[style]
    else:
        raise TypeError(f"style '{style}' provided to ``qml.drawer.use_style`` does not exist.  Available options are {available_styles()}")