from pathlib import Path
from wasabi import MarkdownRenderer, msg
from ..util import load_project_config, working_dir
from .main import PROJECT_FILE, Arg, Opt, app
def project_document(project_dir: Path, output_file: Path, *, no_emoji: bool=False) -> None:
    is_stdout = str(output_file) == '-'
    config = load_project_config(project_dir)
    md = MarkdownRenderer(no_emoji=no_emoji)
    md.add(MARKER_START.format(tag='WEASEL'))
    title = config.get('title')
    description = config.get('description')
    md.add(md.title(1, f'Weasel Project{(f': {title}' if title else '')}', '🪐'))
    if description:
        md.add(description)
    md.add(md.title(2, PROJECT_FILE, '📋'))
    md.add(INTRO_PROJECT)
    cmds = config.get('commands', [])
    data = [(md.code(cmd['name']), cmd.get('help', '')) for cmd in cmds]
    if data:
        md.add(md.title(3, 'Commands', '⏯'))
        md.add(INTRO_COMMANDS)
        md.add(md.table(data, ['Command', 'Description']))
    wfs = config.get('workflows', {}).items()
    data = [(md.code(n), ' &rarr; '.join((md.code(w) for w in stp))) for n, stp in wfs]
    if data:
        md.add(md.title(3, 'Workflows', '⏭'))
        md.add(INTRO_WORKFLOWS)
        md.add(md.table(data, ['Workflow', 'Steps']))
    assets = config.get('assets', [])
    data = []
    for a in assets:
        source = 'Git' if a.get('git') else 'URL' if a.get('url') else 'Local'
        dest_path = a['dest']
        dest = md.code(dest_path)
        if source == 'Local':
            with working_dir(project_dir) as p:
                if (p / dest_path).exists():
                    dest = md.link(dest, dest_path)
        data.append((dest, source, a.get('description', '')))
    if data:
        md.add(md.title(3, 'Assets', '🗂'))
        md.add(INTRO_ASSETS)
        md.add(md.table(data, ['File', 'Source', 'Description']))
    md.add(MARKER_END.format(tag='WEASEL'))
    if is_stdout:
        print(md.text)
    else:
        content = md.text
        if output_file.exists():
            with output_file.open('r', encoding='utf8') as f:
                existing = f.read()
            for marker_tag in MARKER_TAGS:
                if MARKER_IGNORE.format(tag=marker_tag) in existing:
                    msg.warn('Found ignore marker in existing file: skipping', output_file)
                    return
            marker_tag_found = False
            for marker_tag in MARKER_TAGS:
                markers = {'start': MARKER_START.format(tag=marker_tag), 'end': MARKER_END.format(tag=marker_tag)}
                if markers['start'] in existing and markers['end'] in existing:
                    marker_tag_found = True
                    msg.info('Found existing file: only replacing auto-generated docs')
                    before = existing.split(markers['start'])[0]
                    after = existing.split(markers['end'])[1]
                    content = f'{before}{content}{after}'
                    break
            if not marker_tag_found:
                msg.warn('Replacing existing file')
        with output_file.open('w', encoding='utf8') as f:
            f.write(content)
        msg.good('Saved project documentation', output_file)