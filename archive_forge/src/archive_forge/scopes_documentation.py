from taskflow.engines.action_engine import compiler as co
from taskflow.engines.action_engine import traversal as tr
from taskflow import logging
Iterates over the visible scopes.

        How this works is the following:

        We first grab all the predecessors of the given atom (lets call it
        ``Y``) by using the :py:class:`~.compiler.Compilation` execution
        graph (and doing a reverse breadth-first expansion to gather its
        predecessors), this is useful since we know they *always* will
        exist (and execute) before this atom but it does not tell us the
        corresponding scope *level* (flow, nested flow...) that each
        predecessor was created in, so we need to find this information.

        For that information we consult the location of the atom ``Y`` in the
        :py:class:`~.compiler.Compilation` hierarchy/tree. We lookup in a
        reverse order the parent ``X`` of ``Y`` and traverse backwards from
        the index in the parent where ``Y`` exists to all siblings (and
        children of those siblings) in ``X`` that we encounter in this
        backwards search (if a sibling is a flow itself, its atom(s)
        will be recursively expanded and included). This collection will
        then be assumed to be at the same scope. This is what is called
        a *potential* single scope, to make an *actual* scope we remove the
        items from the *potential* scope that are **not** predecessors
        of ``Y`` to form the *actual* scope which we then yield back.

        Then for additional scopes we continue up the tree, by finding the
        parent of ``X`` (lets call it ``Z``) and perform the same operation,
        going through the children in a reverse manner from the index in
        parent ``Z`` where ``X`` was located. This forms another *potential*
        scope which we provide back as an *actual* scope after reducing the
        potential set to only include predecessors previously gathered. We
        then repeat this process until we no longer have any parent
        nodes (aka we have reached the top of the tree) or we run out of
        predecessors.
        