from __future__ import annotations
from typing import TYPE_CHECKING, Iterator, Mapping, Optional, Sequence
from rdflib.graph import Graph
from rdflib.plugins.sparql.evaluate import evalBGP, evalPart
from rdflib.plugins.sparql.evalutils import _fillTemplate, _join
from rdflib.plugins.sparql.parserutils import CompValue
from rdflib.plugins.sparql.sparql import FrozenDict, QueryContext, Update
from rdflib.term import Identifier, URIRef, Variable
def evalModify(ctx: QueryContext, u: CompValue) -> None:
    originalctx = ctx
    dg: Optional[Graph]
    if u.using:
        otherDefault = False
        for d in u.using:
            if d.default:
                if not otherDefault:
                    dg = Graph()
                    ctx = ctx.pushGraph(dg)
                    otherDefault = True
                ctx.load(d.default, default=True)
            elif d.named:
                g = d.named
                ctx.load(g, default=False)
    if not u.using and u.withClause:
        g = ctx.dataset.get_context(u.withClause)
        ctx = ctx.pushGraph(g)
    res = evalPart(ctx, u.where)
    if u.using:
        if otherDefault:
            ctx = originalctx
        if u.withClause:
            g = ctx.dataset.get_context(u.withClause)
            ctx = ctx.pushGraph(g)
    for c in res:
        dg = ctx.graph
        if u.delete:
            dg -= _fillTemplate(u.delete.triples, c)
            for g, q in u.delete.quads.items():
                cg = ctx.dataset.get_context(c.get(g))
                cg -= _fillTemplate(q, c)
        if u.insert:
            dg += _fillTemplate(u.insert.triples, c)
            for g, q in u.insert.quads.items():
                cg = ctx.dataset.get_context(c.get(g))
                cg += _fillTemplate(q, c)