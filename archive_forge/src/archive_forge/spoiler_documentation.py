import re
A mistune plugin to support block and inline spoiler. The
    syntax is inspired by stackexchange:

    .. code-block:: text

        Block level spoiler looks like block quote, but with `>!`:

        >! this is spoiler
        >!
        >! the content will be hidden

        Inline spoiler is surrounded by `>!` and `!<`, such as >! hide me !<.

    :param md: Markdown instance
    