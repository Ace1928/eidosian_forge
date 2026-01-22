import operator
from .. import errors, ui
from ..i18n import gettext
from ..revision import NULL_REVISION
from ..trace import mutter
def make_fetch_spec(self):
    """Build a SearchResult or PendingAncestryResult or etc."""
    from . import vf_search
    if self.target_repo_kind is None or self.source_repo is None:
        raise AssertionError('Incomplete FetchSpecFactory: {!r}'.format(self.__dict__))
    if len(self._explicit_rev_ids) == 0 and self.source_branch is None:
        if self.limit is not None:
            raise NotImplementedError('limit is only supported with a source branch set')
        if self.target_repo_kind == TargetRepoKinds.EMPTY:
            return vf_search.EverythingResult(self.source_repo)
        else:
            return vf_search.EverythingNotInOther(self.target_repo, self.source_repo).execute()
    heads_to_fetch = set(self._explicit_rev_ids)
    if self.source_branch is not None:
        must_fetch, if_present_fetch = self.source_branch.heads_to_fetch()
        if self.source_branch_stop_revision_id is not None:
            must_fetch.discard(self.source_branch.last_revision())
            must_fetch.add(self.source_branch_stop_revision_id)
        heads_to_fetch.update(must_fetch)
    else:
        if_present_fetch = set()
    if self.target_repo_kind == TargetRepoKinds.EMPTY:
        all_heads = heads_to_fetch.union(if_present_fetch)
        ret = vf_search.PendingAncestryResult(all_heads, self.source_repo)
        if self.limit is not None:
            graph = self.source_repo.get_graph()
            topo_order = list(graph.iter_topo_order(ret.get_keys()))
            result_set = topo_order[:self.limit]
            ret = self.source_repo.revision_ids_to_search_result(result_set)
        return ret
    else:
        return vf_search.NotInOtherForRevs(self.target_repo, self.source_repo, required_ids=heads_to_fetch, if_present_ids=if_present_fetch, limit=self.limit).execute()