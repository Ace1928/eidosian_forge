from typing import (
from .pathspec import (
from .pattern import (
from .patterns.gitwildmatch import (
from .util import (

		Check the file against the patterns.

		.. NOTE:: Subclasses of :class:`~pathspec.pathspec.PathSpec` may override
		   this method as an instance method. It does not have to be a static
		   method. The signature for this method is subject to change.

		*patterns* (:class:`~collections.abc.Iterable`) yields each indexed pattern
		(:class:`tuple`) which contains the pattern index (:class:`int`) and actual
		pattern (:class:`~pathspec.pattern.Pattern`).

		*file* (:class:`str`) is the normalized file path to be matched against
		*patterns*.

		Returns a :class:`tuple` containing whether to include *file* (:class:`bool`
		or :data:`None`), and the index of the last matched pattern (:class:`int` or
		:data:`None`).
		