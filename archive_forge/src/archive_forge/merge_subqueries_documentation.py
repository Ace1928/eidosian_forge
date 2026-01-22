from collections import defaultdict
from sqlglot import expressions as exp
from sqlglot.helper import find_new_name
from sqlglot.optimizer.scope import Scope, traverse_scope

        All columns from the inner select in the ON clause must be from the first FROM table.

        That is, this can be merged:
            SELECT * FROM x JOIN (SELECT y.a AS a FROM y JOIN z) AS q ON x.a = q.a
                                         ^^^           ^
        But this can't:
            SELECT * FROM x JOIN (SELECT z.a AS a FROM y JOIN z) AS q ON x.a = q.a
                                         ^^^                  ^
        