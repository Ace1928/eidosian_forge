from __future__ import annotations
from typing import Any
from typing import TYPE_CHECKING
from .base import SchemaEventTarget
from .. import event
def column_reflect(self, inspector: Inspector, table: Table, column_info: ReflectedColumn) -> None:
    """Called for each unit of 'column info' retrieved when
        a :class:`_schema.Table` is being reflected.

        This event is most easily used by applying it to a specific
        :class:`_schema.MetaData` instance, where it will take effect for
        all :class:`_schema.Table` objects within that
        :class:`_schema.MetaData` that undergo reflection::

            metadata = MetaData()

            @event.listens_for(metadata, 'column_reflect')
            def receive_column_reflect(inspector, table, column_info):
                # receives for all Table objects that are reflected
                # under this MetaData


            # will use the above event hook
            my_table = Table("my_table", metadata, autoload_with=some_engine)


        .. versionadded:: 1.4.0b2 The :meth:`_events.DDLEvents.column_reflect`
           hook may now be applied to a :class:`_schema.MetaData` object as
           well as the :class:`_schema.MetaData` class itself where it will
           take place for all :class:`_schema.Table` objects associated with
           the targeted :class:`_schema.MetaData`.

        It may also be applied to the :class:`_schema.Table` class across
        the board::

            from sqlalchemy import Table

            @event.listens_for(Table, 'column_reflect')
            def receive_column_reflect(inspector, table, column_info):
                # receives for all Table objects that are reflected

        It can also be applied to a specific :class:`_schema.Table` at the
        point that one is being reflected using the
        :paramref:`_schema.Table.listeners` parameter::

            t1 = Table(
                "my_table",
                autoload_with=some_engine,
                listeners=[
                    ('column_reflect', receive_column_reflect)
                ]
            )

        The dictionary of column information as returned by the
        dialect is passed, and can be modified.  The dictionary
        is that returned in each element of the list returned
        by :meth:`.reflection.Inspector.get_columns`:

            * ``name`` - the column's name, is applied to the
              :paramref:`_schema.Column.name` parameter

            * ``type`` - the type of this column, which should be an instance
              of :class:`~sqlalchemy.types.TypeEngine`, is applied to the
              :paramref:`_schema.Column.type` parameter

            * ``nullable`` - boolean flag if the column is NULL or NOT NULL,
              is applied to the :paramref:`_schema.Column.nullable` parameter

            * ``default`` - the column's server default value.  This is
              normally specified as a plain string SQL expression, however the
              event can pass a :class:`.FetchedValue`, :class:`.DefaultClause`,
              or :func:`_expression.text` object as well.  Is applied to the
              :paramref:`_schema.Column.server_default` parameter

        The event is called before any action is taken against
        this dictionary, and the contents can be modified; the following
        additional keys may be added to the dictionary to further modify
        how the :class:`_schema.Column` is constructed:


            * ``key`` - the string key that will be used to access this
              :class:`_schema.Column` in the ``.c`` collection; will be applied
              to the :paramref:`_schema.Column.key` parameter. Is also used
              for ORM mapping.  See the section
              :ref:`mapper_automated_reflection_schemes` for an example.

            * ``quote`` - force or un-force quoting on the column name;
              is applied to the :paramref:`_schema.Column.quote` parameter.

            * ``info`` - a dictionary of arbitrary data to follow along with
              the :class:`_schema.Column`, is applied to the
              :paramref:`_schema.Column.info` parameter.

        :func:`.event.listen` also accepts the ``propagate=True``
        modifier for this event; when True, the listener function will
        be established for any copies made of the target object,
        i.e. those copies that are generated when
        :meth:`_schema.Table.to_metadata` is used.

        .. seealso::

            :ref:`mapper_automated_reflection_schemes` -
            in the ORM mapping documentation

            :ref:`automap_intercepting_columns` -
            in the :ref:`automap_toplevel` documentation

            :ref:`metadata_reflection_dbagnostic_types` - in
            the :ref:`metadata_reflection_toplevel` documentation

        """