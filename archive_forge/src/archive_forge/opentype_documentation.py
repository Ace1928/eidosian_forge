Create ASN.1 type map indexed by a value

    The *DefinedBy* object models the ASN.1 *DEFINED BY* clause which maps
    values to ASN.1 types in the context of the ASN.1 SEQUENCE/SET type.

    OpenType objects are duck-type a read-only Python :class:`dict` objects,
    however the passed `typeMap` is stored by reference.

    Parameters
    ----------
    name: :py:class:`str`
        Field name

    typeMap: :py:class:`dict`
        A map of value->ASN.1 type. It's stored by reference and can be
        mutated later to register new mappings.

    Examples
    --------
    .. code-block:: python

        openType = OpenType(
            'id',
            {1: Integer(),
             2: OctetString()}
        )
        Sequence(
            componentType=NamedTypes(
                NamedType('id', Integer()),
                NamedType('blob', Any(), openType=openType)
            )
        )
    