from yaql.language import specs
@specs.parameter('left', type(None), nullable=True)
@specs.parameter('right', nullable=False)
@specs.name('#operator_>')
def null_gt_right(left, right):
    """:yaql:operator >

    Returns false. This function is called when left is null and right
    is not.

    :signature: left > right
    :arg left: left operand
    :argType left: null
    :arg right: right operand
    :argType right: not null
    :returnType: boolean

    .. code:

        yaql> null > 2
        false
    """
    return False