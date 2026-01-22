import os
import sys
import re
def expand_sub(substr, names):
    substr = substr.replace('\\>', '@rightarrow@')
    substr = substr.replace('\\<', '@leftarrow@')
    lnames = find_repl_patterns(substr)
    substr = named_re.sub('<\\1>', substr)

    def listrepl(mobj):
        thelist = conv(mobj.group(1).replace('\\,', '@comma@'))
        if template_name_re.match(thelist):
            return '<%s>' % thelist
        name = None
        for key in lnames.keys():
            if lnames[key] == thelist:
                name = key
        if name is None:
            name = unique_key(lnames)
            lnames[name] = thelist
        return '<%s>' % name
    substr = list_re.sub(listrepl, substr)
    numsubs = None
    base_rule = None
    rules = {}
    for r in template_re.findall(substr):
        if r not in rules:
            thelist = lnames.get(r, names.get(r, None))
            if thelist is None:
                raise ValueError('No replicates found for <%s>' % r)
            if r not in names and (not thelist.startswith('_')):
                names[r] = thelist
            rule = [i.replace('@comma@', ',') for i in thelist.split(',')]
            num = len(rule)
            if numsubs is None:
                numsubs = num
                rules[r] = rule
                base_rule = r
            elif num == numsubs:
                rules[r] = rule
            else:
                print('Mismatch in number of replacements (base <%s=%s>) for <%s=%s>. Ignoring.' % (base_rule, ','.join(rules[base_rule]), r, thelist))
    if not rules:
        return substr

    def namerepl(mobj):
        name = mobj.group(1)
        return rules.get(name, (k + 1) * [name])[k]
    newstr = ''
    for k in range(numsubs):
        newstr += template_re.sub(namerepl, substr) + '\n\n'
    newstr = newstr.replace('@rightarrow@', '>')
    newstr = newstr.replace('@leftarrow@', '<')
    return newstr