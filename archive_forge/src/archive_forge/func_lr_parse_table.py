import re
import types
import sys
import os.path
import inspect
import base64
import warnings
def lr_parse_table(self):
    Productions = self.grammar.Productions
    Precedence = self.grammar.Precedence
    goto = self.lr_goto
    action = self.lr_action
    log = self.log
    actionp = {}
    log.info('Parsing method: %s', self.lr_method)
    C = self.lr0_items()
    if self.lr_method == 'LALR':
        self.add_lalr_lookaheads(C)
    st = 0
    for I in C:
        actlist = []
        st_action = {}
        st_actionp = {}
        st_goto = {}
        log.info('')
        log.info('state %d', st)
        log.info('')
        for p in I:
            log.info('    (%d) %s', p.number, p)
        log.info('')
        for p in I:
            if p.len == p.lr_index + 1:
                if p.name == "S'":
                    st_action['$end'] = 0
                    st_actionp['$end'] = p
                else:
                    if self.lr_method == 'LALR':
                        laheads = p.lookaheads[st]
                    else:
                        laheads = self.grammar.Follow[p.name]
                    for a in laheads:
                        actlist.append((a, p, 'reduce using rule %d (%s)' % (p.number, p)))
                        r = st_action.get(a)
                        if r is not None:
                            if r > 0:
                                sprec, slevel = Precedence.get(a, ('right', 0))
                                rprec, rlevel = Productions[p.number].prec
                                if slevel < rlevel or (slevel == rlevel and rprec == 'left'):
                                    st_action[a] = -p.number
                                    st_actionp[a] = p
                                    if not slevel and (not rlevel):
                                        log.info('  ! shift/reduce conflict for %s resolved as reduce', a)
                                        self.sr_conflicts.append((st, a, 'reduce'))
                                    Productions[p.number].reduced += 1
                                elif slevel == rlevel and rprec == 'nonassoc':
                                    st_action[a] = None
                                elif not rlevel:
                                    log.info('  ! shift/reduce conflict for %s resolved as shift', a)
                                    self.sr_conflicts.append((st, a, 'shift'))
                            elif r < 0:
                                oldp = Productions[-r]
                                pp = Productions[p.number]
                                if oldp.line > pp.line:
                                    st_action[a] = -p.number
                                    st_actionp[a] = p
                                    chosenp, rejectp = (pp, oldp)
                                    Productions[p.number].reduced += 1
                                    Productions[oldp.number].reduced -= 1
                                else:
                                    chosenp, rejectp = (oldp, pp)
                                self.rr_conflicts.append((st, chosenp, rejectp))
                                log.info('  ! reduce/reduce conflict for %s resolved using rule %d (%s)', a, st_actionp[a].number, st_actionp[a])
                            else:
                                raise LALRError('Unknown conflict in state %d' % st)
                        else:
                            st_action[a] = -p.number
                            st_actionp[a] = p
                            Productions[p.number].reduced += 1
            else:
                i = p.lr_index
                a = p.prod[i + 1]
                if a in self.grammar.Terminals:
                    g = self.lr0_goto(I, a)
                    j = self.lr0_cidhash.get(id(g), -1)
                    if j >= 0:
                        actlist.append((a, p, 'shift and go to state %d' % j))
                        r = st_action.get(a)
                        if r is not None:
                            if r > 0:
                                if r != j:
                                    raise LALRError('Shift/shift conflict in state %d' % st)
                            elif r < 0:
                                sprec, slevel = Precedence.get(a, ('right', 0))
                                rprec, rlevel = Productions[st_actionp[a].number].prec
                                if slevel > rlevel or (slevel == rlevel and rprec == 'right'):
                                    Productions[st_actionp[a].number].reduced -= 1
                                    st_action[a] = j
                                    st_actionp[a] = p
                                    if not rlevel:
                                        log.info('  ! shift/reduce conflict for %s resolved as shift', a)
                                        self.sr_conflicts.append((st, a, 'shift'))
                                elif slevel == rlevel and rprec == 'nonassoc':
                                    st_action[a] = None
                                elif not slevel and (not rlevel):
                                    log.info('  ! shift/reduce conflict for %s resolved as reduce', a)
                                    self.sr_conflicts.append((st, a, 'reduce'))
                            else:
                                raise LALRError('Unknown conflict in state %d' % st)
                        else:
                            st_action[a] = j
                            st_actionp[a] = p
        _actprint = {}
        for a, p, m in actlist:
            if a in st_action:
                if p is st_actionp[a]:
                    log.info('    %-15s %s', a, m)
                    _actprint[a, m] = 1
        log.info('')
        not_used = 0
        for a, p, m in actlist:
            if a in st_action:
                if p is not st_actionp[a]:
                    if not (a, m) in _actprint:
                        log.debug('  ! %-15s [ %s ]', a, m)
                        not_used = 1
                        _actprint[a, m] = 1
        if not_used:
            log.debug('')
        nkeys = {}
        for ii in I:
            for s in ii.usyms:
                if s in self.grammar.Nonterminals:
                    nkeys[s] = None
        for n in nkeys:
            g = self.lr0_goto(I, n)
            j = self.lr0_cidhash.get(id(g), -1)
            if j >= 0:
                st_goto[n] = j
                log.info('    %-30s shift and go to state %d', n, j)
        action[st] = st_action
        actionp[st] = st_actionp
        goto[st] = st_goto
        st += 1