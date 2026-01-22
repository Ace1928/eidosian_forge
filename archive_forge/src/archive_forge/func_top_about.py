def top_about(root, options, state):
    """
    @param root: a DOM node for the top level element
    @param options: invocation options
    @type options: L{Options<pyRdfa.options>}
    @param state: top level execution state
    @type state: L{State<pyRdfa.state>}
    """

    def set_about(node):
        if has_one_of_attributes(node, 'rel', 'rev'):
            if not has_one_of_attributes(top, 'about', 'src'):
                node.setAttribute('about', '')
        elif not has_one_of_attributes(node, 'href', 'resource', 'about', 'src'):
            node.setAttribute('about', '')
    from ..host import HostLanguage
    from ..utils import has_one_of_attributes
    if not has_one_of_attributes(root, 'about'):
        if has_one_of_attributes(root, 'resource', 'href', 'src'):
            if has_one_of_attributes(root, 'rel', 'rev', 'property'):
                root.setAttribute('about', '')
        else:
            root.setAttribute('about', '')
    if options.host_language in [HostLanguage.xhtml, HostLanguage.html5, HostLanguage.xhtml5]:
        if state.rdfa_version >= '1.1':
            pass
        else:
            for top in root.getElementsByTagName('head'):
                if not has_one_of_attributes(top, 'href', 'resource', 'about', 'src'):
                    set_about(top)
            for top in root.getElementsByTagName('body'):
                if not has_one_of_attributes(top, 'href', 'resource', 'about', 'src'):
                    set_about(top)