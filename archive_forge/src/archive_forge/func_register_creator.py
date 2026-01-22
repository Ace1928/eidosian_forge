import copy
from typing import Any, Callable, Dict, List, Optional, no_type_check
from triad import ParamDict
from triad.collections import Schema
from triad.utils.assertion import assert_or_throw
from triad.utils.convert import get_caller_global_local_vars, to_function, to_instance
from triad.utils.hash import to_uuid
from fugue._utils.interfaceless import parse_output_schema_from_comment
from fugue._utils.registry import fugue_plugin
from fugue.dataframe import DataFrame
from fugue.dataframe.function_wrapper import DataFrameFunctionWrapper
from fugue.exceptions import FugueInterfacelessError
from fugue.extensions.creator.creator import Creator
from .._utils import load_namespace_extensions
def register_creator(alias: str, obj: Any, on_dup: int=ParamDict.OVERWRITE) -> None:
    """Register creator with an alias. This is a simplified version of
    :func:`~.parse_creator`

    :param alias: alias of the creator
    :param obj: the object that can be converted to
        :class:`~fugue.extensions.creator.creator.Creator`
    :param on_dup: see :meth:`triad.collections.dict.ParamDict.update`
        , defaults to ``ParamDict.OVERWRITE``

    .. tip::

        Registering an extension with an alias is particularly useful for projects
        such as libraries. This is because by using alias, users don't have to
        import the specific extension, or provide the full path of the extension.
        It can make user's code less dependent and easy to understand.

    .. admonition:: New Since
        :class: hint

        **0.6.0**

    .. seealso::

        Please read
        :doc:`Creator Tutorial <tutorial:tutorials/extensions/creator>`

    .. admonition:: Examples

        Here is an example how you setup your project so your users can
        benefit from this feature. Assume your project name is ``pn``

        The creator implementation in file ``pn/pn/creators.py``

        .. code-block:: python

            import pandas import pd

            def my_creator() -> pd.DataFrame:
                return pd.DataFrame()

        Then in ``pn/pn/__init__.py``

        .. code-block:: python

            from .creators import my_creator
            from fugue import register_creator

            def register_extensions():
                register_creator("mc", my_creator)
                # ... register more extensions

            register_extensions()

        In users code:

        .. code-block:: python

            import pn  # register_extensions will be called
            from fugue import FugueWorkflow

            dag = FugueWorkflow()
            dag.create("mc").show()  # use my_creator by alias
            dag.run()
    """
    _CREATOR_REGISTRY.update({alias: obj}, on_dup=on_dup)