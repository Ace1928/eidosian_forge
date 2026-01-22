import threading
Evaluate the ops with the given input.

        When this function is called, the default session will have the
        graph defined by a previous call to `initialize_graph`. This
        function should evaluate any ops necessary to compute the result
        of the query for the given *args and **kwargs, likely returning
        the result of a call to `some_op.eval(...)`.
        