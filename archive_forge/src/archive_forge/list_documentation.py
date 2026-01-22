import logging
from cmakelang.lex import TokenType
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import get_first_semantic_token

  Reading
    list(LENGTH <list> <out-var>)
    list(GET <list> <element index> [<index> ...] <out-var>)
    list(JOIN <list> <glue> <out-var>)
    list(SUBLIST <list> <begin> <length> <out-var>)

  Search
    list(FIND <list> <value> <out-var>)

  Modification
    list(APPEND <list> [<element>...])
    list(FILTER <list> {INCLUDE | EXCLUDE} REGEX <regex>)
    list(INSERT <list> <index> [<element>...])
    list(POP_BACK <list> [<out-var>...])
    list(POP_FRONT <list> [<out-var>...])
    list(PREPEND <list> [<element>...])
    list(REMOVE_ITEM <list> <value>...)
    list(REMOVE_AT <list> <index>...)
    list(REMOVE_DUPLICATES <list>)
    list(TRANSFORM <list> <ACTION> [...])

  Ordering
    list(REVERSE <list>)
    list(SORT <list> [...])

  TRANSFORM actions:
    list(TRANSFORM <list> <APPEND|PREPEND> <value> ...)
    list(TRANSFORM <list> <TOLOWER|TOUPPER> ...)
    list(TRANSFORM <list> STRIP ...)
    list(TRANSFORM <list> GENEX_STRIP ...)
    list(TRANSFORM <list> REPLACE <regular_expression>
                                  <replace_expression> ...)

  TRANSFORM selectors
    list(TRANSFORM <list> <ACTION> AT <index> [<index> ...] ...)
    list(TRANSFORM <list> <ACTION> FOR <start> <stop> [<step>] ...)
    list(TRANSFORM <list> <ACTION> REGEX <regular_expression> ...)

  :see: https://cmake.org/cmake/help/latest/command/list.html
  