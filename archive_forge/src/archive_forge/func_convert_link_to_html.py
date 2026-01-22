import pyparsing as pp
def convert_link_to_html(s, l, t):
    link_text, url = t._skipped
    t['link_text'] = wiki_markup.transformString(link_text)
    t['url'] = url
    return '<A href="{url}">{link_text}</A>'.format_map(t)