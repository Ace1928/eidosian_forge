import copy
from typing import Any, Callable, Dict, List, Optional, no_type_check
from triad import ParamDict, to_uuid
from triad.utils.convert import get_caller_global_local_vars, to_function, to_instance
from fugue._utils.registry import fugue_plugin
from fugue.dataframe import DataFrames
from fugue.dataframe.function_wrapper import DataFrameFunctionWrapper
from fugue.exceptions import FugueInterfacelessError
from fugue.extensions._utils import (
from fugue.extensions.outputter.outputter import Outputter
def register_outputter(alias: str, obj: Any, on_dup: int=ParamDict.OVERWRITE) -> None:
    """Register outputter with an alias.

    :param alias: alias of the processor
    :param obj: the object that can be converted to
        :class:`~fugue.extensions.outputter.outputter.Outputter`
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
        :doc:`Outputter Tutorial <tutorial:tutorials/extensions/outputter>`

    .. admonition:: Examples

        Here is an example how you setup your project so your users can
        benefit from this feature. Assume your project name is ``pn``

        The processor implementation in file ``pn/pn/outputters.py``

        .. code-block:: python

            from fugue import DataFrame

            def my_outputter(df:DataFrame) -> None:
                print(df)

        Then in ``pn/pn/__init__.py``

        .. code-block:: python

            from .outputters import my_outputter
            from fugue import register_outputter

            def register_extensions():
                register_outputter("mo", my_outputter)
                # ... register more extensions

            register_extensions()

        In users code:

        .. code-block:: python

            import pn  # register_extensions will be called
            from fugue import FugueWorkflow

            dag = FugueWorkflow()
            # use my_outputter by alias
            dag.df([[0]],"a:int").output("mo")
            dag.run()
    """
    _OUTPUTTER_REGISTRY.update({alias: obj}, on_dup=on_dup)