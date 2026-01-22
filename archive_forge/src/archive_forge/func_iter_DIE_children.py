from bisect import bisect_right
from .die import DIE
from ..common.utils import dwarf_assert
def iter_DIE_children(self, die):
    """ Given a DIE, yields either its children, without null DIE list
            terminator, or nothing, if that DIE has no children.

            The null DIE terminator is saved in that DIE when iteration ended.
        """
    if not die.has_children:
        return
    cur_offset = die.offset + die.size
    while True:
        child = self._get_cached_DIE(cur_offset)
        child.set_parent(die)
        if child.is_null():
            die._terminator = child
            return
        yield child
        if not child.has_children:
            cur_offset += child.size
        elif 'DW_AT_sibling' in child.attributes:
            sibling = child.attributes['DW_AT_sibling']
            if sibling.form in ('DW_FORM_ref1', 'DW_FORM_ref2', 'DW_FORM_ref4', 'DW_FORM_ref8', 'DW_FORM_ref', 'DW_FORM_ref_udata'):
                cur_offset = sibling.value + self.cu_offset
            elif sibling.form == 'DW_FORM_ref_addr':
                cur_offset = sibling.value
            else:
                raise NotImplementedError('sibling in form %s' % sibling.form)
        else:
            if child._terminator is None:
                for _ in self.iter_DIE_children(child):
                    pass
            cur_offset = child._terminator.offset + child._terminator.size