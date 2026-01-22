import collections
from collections import defaultdict
from .node import Node, Argument, Target, map_arg, _type_repr, _get_qualified_name
import torch.utils._pytree as pytree
from . import _pytree as fx_pytree
from ._compatibility import compatibility
import contextlib
from typing import TYPE_CHECKING, Callable, Any, List, Dict, NamedTuple, Optional, Tuple, Set, FrozenSet, Type
from dataclasses import dataclass
from contextlib import contextmanager
import copy
import enum
import torch
import keyword
import re
import builtins
import math
import warnings
import inspect
@compatibility(is_backward_compatible=False)
def on_generate_code(self, make_transformer: Callable[[Optional[TransformCodeFunc]], TransformCodeFunc]):
    """Register a transformer function when python code is generated

        Args:
            make_transformer (Callable[[Optional[TransformCodeFunc]], TransformCodeFunc]):
                a function that returns a code transformer to be registered.
                This function is called by `on_generate_code` to obtain the
                code transformer.

                This function is also given as its input the currently
                registered code transformer (or None if nothing is registered),
                in case it is not desirable to overwrite it. This is useful to
                chain code transformers together.

        Returns:
            a context manager that when used in a `with` statement, to automatically
            restore the previously registered code transformer.

        Example:

        .. code-block:: python


            gm: fx.GraphModule = ...

            # This is a code transformer we want to register. This code
            # transformer prepends a pdb import and trace statement at the very
            # beginning of the generated torch.fx code to allow for manual
            # debugging with the PDB library.
            def insert_pdb(body):
                return ["import pdb; pdb.set_trace()\\n", *body]

            # Registers `insert_pdb`, and overwrites the current registered
            # code transformer (given by `_` to the lambda):
            gm.graph.on_generate_code(
                lambda _: insert_pdb
            )

            # Or alternatively, registers a code transformer which first
            # runs `body` through existing registered transformer, then
            # through `insert_pdb`:
            gm.graph.on_generate_code(
                lambda current_trans: (
                    lambda body: insert_pdb(
                        current_trans(body) if current_trans
                        else body
                    )
                )
            )

            gm.recompile()
            gm(*inputs)  # drops into pdb


        This function can also be used as a context manager, with the benefit to
        automatically restores the previously registered code transformer:

        .. code-block:: python

            # ... continue from previous example

            with gm.graph.on_generate_code(lambda _: insert_pdb):
                # do more stuff with `gm`...
                gm.recompile()
                gm(*inputs)  # drops into pdb

            # now previous code transformer is restored (but `gm`'s code with pdb
            # remains - that means you can run `gm` with pdb here too, until you
            # run next `recompile()`).
        """
    on_gen_code_old = self._codegen._body_transformer
    self._codegen._body_transformer = make_transformer(on_gen_code_old)

    @contextlib.contextmanager
    def on_generate_code_context_manager():
        try:
            yield
        finally:
            self._codegen._body_transformer = on_gen_code_old
    return on_generate_code_context_manager()