import os
import subprocess
import sys
from debugpy import adapter, common
from debugpy.common import log, messaging, sockets
from debugpy.adapter import components, servers, sessions
def spawn_debuggee(session, start_request, python, launcher_path, adapter_host, args, shell_expand_args, cwd, console, console_title, sudo):
    global listener
    cmdline = ['sudo', '-E'] if sudo else []
    cmdline += python
    cmdline += [launcher_path]
    env = {}
    arguments = dict(start_request.arguments)
    if not session.no_debug:
        _, arguments['port'] = servers.listener.getsockname()
        arguments['adapterAccessToken'] = adapter.access_token

    def on_launcher_connected(sock):
        listener.close()
        stream = messaging.JsonIOStream.from_socket(sock)
        Launcher(session, stream)
    try:
        listener = sockets.serve('Launcher', on_launcher_connected, adapter_host, backlog=1)
    except Exception as exc:
        raise start_request.cant_handle("{0} couldn't create listener socket for launcher: {1}", session, exc)
    sessions.report_sockets()
    try:
        launcher_host, launcher_port = listener.getsockname()
        launcher_addr = launcher_port if launcher_host == '127.0.0.1' else f'{launcher_host}:{launcher_port}'
        cmdline += [str(launcher_addr), '--']
        cmdline += args
        if log.log_dir is not None:
            env[str('DEBUGPY_LOG_DIR')] = log.log_dir
        if log.stderr.levels != {'warning', 'error'}:
            env[str('DEBUGPY_LOG_STDERR')] = str(' '.join(log.stderr.levels))
        if console == 'internalConsole':
            log.info('{0} spawning launcher: {1!r}', session, cmdline)
            try:
                subprocess.Popen(cmdline, cwd=cwd, env=dict(list(os.environ.items()) + list(env.items())), stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)
            except Exception as exc:
                raise start_request.cant_handle('Failed to spawn launcher: {0}', exc)
        else:
            log.info('{0} spawning launcher via "runInTerminal" request.', session)
            session.client.capabilities.require('supportsRunInTerminalRequest')
            kinds = {'integratedTerminal': 'integrated', 'externalTerminal': 'external'}
            request_args = {'kind': kinds[console], 'title': console_title, 'args': cmdline, 'env': env}
            if cwd is not None:
                request_args['cwd'] = cwd
            if shell_expand_args:
                request_args['argsCanBeInterpretedByShell'] = True
            try:
                session.client.channel.send_request('runInTerminal', request_args)
            except messaging.MessageHandlingError as exc:
                exc.propagate(start_request)
        if not session.wait_for(lambda: session.launcher, timeout=None if sudo else common.PROCESS_SPAWN_TIMEOUT):
            raise start_request.cant_handle('Timed out waiting for launcher to connect')
        try:
            session.launcher.channel.request(start_request.command, arguments)
        except messaging.MessageHandlingError as exc:
            exc.propagate(start_request)
        if not session.wait_for(lambda: session.launcher.pid is not None, timeout=common.PROCESS_SPAWN_TIMEOUT):
            raise start_request.cant_handle('Timed out waiting for "process" event from launcher')
        if session.no_debug:
            return
        conn = servers.wait_for_connection(session, lambda conn: True, timeout=common.PROCESS_SPAWN_TIMEOUT)
        if conn is None:
            raise start_request.cant_handle('Timed out waiting for debuggee to spawn')
        conn.attach_to_session(session)
    finally:
        listener.close()
        listener = None
        sessions.report_sockets()