`uarray` provides functions for generating multimethods that dispatch to
multiple different backends

This should be imported, rather than `_uarray` so that an installed version could
be used instead, if available. This means that users can call
`uarray.set_backend` directly instead of going through SciPy.

