#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx

FORGE_ROOT = Path(__file__).resolve().parent.parent
for extra in (FORGE_ROOT / 'lib', FORGE_ROOT / 'eidos_mcp' / 'src', FORGE_ROOT / 'agent_forge' / 'src'):
    value = str(extra)
    if extra.exists() and value not in sys.path:
        sys.path.insert(0, value)

from eidosian_core.ports import get_service_url
from eidosian_runtime import ForgeRuntimeCoordinator, write_runtime_status
from eidosian_runtime.session_bridge import (
    append_session_event,
    build_session_context,
    ingest_session_content,
    new_session_id,
    render_context_packet,
)

RUNTIME_DIR = FORGE_ROOT / 'data' / 'runtime'
SCHEDULER_STATUS_PATH = RUNTIME_DIR / 'eidos_scheduler_status.json'
PIPELINE_STATUS_PATH = RUNTIME_DIR / 'living_pipeline_status.json'
QWENCHAT_STATUS_PATH = RUNTIME_DIR / 'qwenchat' / 'status.json'
QWENCHAT_HISTORY_PATH = RUNTIME_DIR / 'qwenchat' / 'history.jsonl'


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _eta_hint() -> tuple[str, float | None]:
    pipeline = _load_json(PIPELINE_STATUS_PATH)
    scheduler = _load_json(SCHEDULER_STATUS_PATH)
    phase = str(pipeline.get('phase') or scheduler.get('current_task') or '').strip()
    owner = str((scheduler.get('owner') or '')).strip()
    eta = pipeline.get('eta_seconds')
    if eta is None:
        eta = scheduler.get('next_run_in_seconds')
    try:
        eta_value = None if eta is None else max(0.0, float(eta))
    except Exception:
        eta_value = None
    label = phase or owner or 'current task'
    return label, eta_value


def _wait_message(decision: dict[str, Any], coordinator_payload: dict[str, Any]) -> str:
    owner = str(
        decision.get('exclusive_owner') or decision.get('active_owner') or coordinator_payload.get('owner') or ''
    ).strip()
    task = str(coordinator_payload.get('task') or 'current task').strip()
    label, eta = _eta_hint()
    task_label = label or task or 'current task'
    if eta is None:
        return f"waiting for {owner or 'active worker'} to release {task_label}"
    eta_text = f"{int(round(eta))}s"
    return f"waiting for {owner or 'active worker'} to release {task_label} (eta ~{eta_text})"


def _lease_metadata(model: str) -> dict[str, Any]:
    return {
        'exclusive': True,
        'exclusive_owner': 'qwenchat',
        'mode': 'interactive_chat',
        'model': model,
        'chat_url': get_service_url(
            'ollama_qwen_http', default_port=8938, default_host='127.0.0.1', default_path=''
        ).rstrip('/'),
    }


def _write_qwenchat_status(
    *,
    session_id: str,
    model: str,
    status: str,
    phase: str = '',
    message: str = '',
    **extra: Any,
) -> dict[str, Any]:
    return write_runtime_status(
        QWENCHAT_STATUS_PATH,
        contract='eidos.qwenchat.status.v1',
        component='qwenchat',
        status=status,
        phase=phase,
        message=message,
        history_path=QWENCHAT_HISTORY_PATH,
        session_id=session_id,
        model=model,
        **extra,
    )


def acquire_chat_lease(
    coordinator: ForgeRuntimeCoordinator,
    *,
    model: str,
    poll_interval: float,
    out: Any = sys.stderr,
) -> str:
    owner = 'qwenchat'
    requested_models = [{'family': 'ollama', 'model': model, 'role': 'interactive_chat'}]
    last_message = ''
    while True:
        decision = coordinator.can_allocate(owner=owner, requested_models=requested_models, allow_same_owner=False)
        if decision.get('allowed'):
            coordinator.heartbeat(
                owner=owner,
                task='interactive_chat',
                state='interactive',
                active_models=requested_models,
                metadata=_lease_metadata(model),
            )
            return owner
        payload = coordinator.read()
        message = _wait_message(decision, payload)
        if message != last_message:
            print(f'[qwenchat] {message}', file=out)
            last_message = message
        time.sleep(max(1.0, float(poll_interval)))


def _heartbeat_loop(
    coordinator: ForgeRuntimeCoordinator, *, owner: str, model: str, stop_event: threading.Event
) -> None:
    requested_models = [{'family': 'ollama', 'model': model, 'role': 'interactive_chat'}]
    while not stop_event.wait(5.0):
        coordinator.heartbeat(
            owner=owner,
            task='interactive_chat',
            state='interactive',
            active_models=requested_models,
            metadata=_lease_metadata(model),
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Interactive qwen chat against the dedicated Ollama service.')
    parser.add_argument('--model', default=os.environ.get('EIDOS_QWEN_MODEL', 'qwen3.5:2b'))
    parser.add_argument('--poll-interval', type=float, default=5.0)
    parser.add_argument('--prompt', default='', help='Run a single prompt and exit instead of interactive chat.')
    parser.add_argument('--timeout-sec', type=float, default=1800.0, help='Timeout for prompt/chat turns.')
    parser.add_argument('extra', nargs='*', help='Reserved for future compatibility.')
    return parser.parse_args()


def _context_system_message(model: str, context_text: str) -> str:
    return (
        'You are Eidos in the qwenchat interface. Preserve continuity with recent Eidos sessions, '
        'shared memory, and the unified forge runtime. Be concise, grounded, and operational.\n\n'
        f'Runtime model: {model}\n'
        f'{context_text}'
    )


def _chat_completion(*, base_url: str, model: str, messages: list[dict[str, str]], timeout_sec: float) -> str:
    payload = {
        'model': model,
        'messages': messages,
        'stream': False,
        'options': {'num_predict': 512, 'temperature': 0.2},
        'think': False,
        'keep_alive': '4h',
    }
    with httpx.Client(timeout=max(10.0, float(timeout_sec))) as client:
        response = client.post(f'{base_url}/api/chat', json=payload)
        response.raise_for_status()
        body = response.json()
    message = body.get('message') if isinstance(body, dict) else {}
    return str((message or {}).get('content') or '').strip()


def _interactive_loop(*, base_url: str, model: str, timeout_sec: float, session_id: str) -> int:
    print('[qwenchat] interactive mode. commands: /exit /quit /context', file=sys.stderr)
    history: list[dict[str, str]] = []
    while True:
        try:
            prompt = input('qwen> ').strip()
        except EOFError:
            print('', file=sys.stderr)
            return 0
        if not prompt:
            continue
        if prompt in {'/exit', '/quit'}:
            return 0
        context_payload = build_session_context(interface='qwenchat', query=prompt, session_id=session_id)
        context_text = render_context_packet(context_payload)
        if prompt == '/context':
            print(context_text)
            continue
        append_session_event(
            interface='qwenchat',
            session_id=session_id,
            event_type='prompt',
            summary=prompt[:200],
            metadata={'mode': 'interactive'},
        )
        messages = [
            {'role': 'system', 'content': _context_system_message(model, context_text)},
            *history,
            {'role': 'user', 'content': prompt},
        ]
        try:
            reply = _chat_completion(base_url=base_url, model=model, messages=messages, timeout_sec=timeout_sec)
        except httpx.TimeoutException:
            print(f'[qwenchat] request timed out after {float(timeout_sec):.0f}s', file=sys.stderr)
            continue
        except httpx.HTTPError as exc:
            print(f'[qwenchat] request failed: {exc}', file=sys.stderr)
            continue
        print(reply)
        history.extend([
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': reply},
        ])
        history = history[-12:]
        append_session_event(
            interface='qwenchat',
            session_id=session_id,
            event_type='response',
            summary=reply[:200],
            metadata={'mode': 'interactive'},
        )
        ingest_session_content(prompt, reply)


def main() -> int:
    args = _parse_args()
    coordinator = ForgeRuntimeCoordinator()
    session_id = new_session_id('qwenchat')
    mode = 'oneshot' if args.prompt.strip() else 'interactive'
    append_session_event(
        interface='qwenchat',
        session_id=session_id,
        event_type='start',
        summary=f'qwenchat session started for model {args.model}',
        metadata={'model': str(args.model)},
    )
    _write_qwenchat_status(
        session_id=session_id,
        model=str(args.model),
        status='running',
        phase='waiting_for_lease',
        message='waiting for interactive model lease',
        mode=mode,
    )
    owner = acquire_chat_lease(coordinator, model=str(args.model), poll_interval=float(args.poll_interval))
    stop_event = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat_loop,
        args=(coordinator,),
        kwargs={'owner': owner, 'model': str(args.model), 'stop_event': stop_event},
        daemon=True,
    )
    heartbeat.start()

    base_url = get_service_url('ollama_qwen_http', default_port=8938, default_host='127.0.0.1', default_path='').rstrip(
        '/'
    )
    print(f'[qwenchat] using {base_url} model={args.model}', file=sys.stderr)
    _write_qwenchat_status(
        session_id=session_id,
        model=str(args.model),
        status='running',
        phase='ready',
        message='interactive model lease acquired',
        mode=mode,
        base_url=base_url,
    )

    return_code = 1
    try:
        if args.prompt.strip():
            prompt = str(args.prompt)
            _write_qwenchat_status(
                session_id=session_id,
                model=str(args.model),
                status='running',
                phase='model_request',
                message='executing one-shot prompt',
                mode=mode,
                base_url=base_url,
            )
            context_payload = build_session_context(interface='qwenchat', query=prompt, session_id=session_id)
            context_text = render_context_packet(context_payload)
            append_session_event(
                interface='qwenchat',
                session_id=session_id,
                event_type='prompt',
                summary=prompt[:200],
                metadata={'mode': 'oneshot'},
            )
            try:
                reply = _chat_completion(
                    base_url=base_url,
                    model=str(args.model),
                    messages=[
                        {'role': 'system', 'content': _context_system_message(str(args.model), context_text)},
                        {'role': 'user', 'content': prompt},
                    ],
                    timeout_sec=float(args.timeout_sec),
                )
                print(reply)
                append_session_event(
                    interface='qwenchat',
                    session_id=session_id,
                    event_type='response',
                    summary=reply[:200],
                    metadata={'mode': 'oneshot'},
                )
                ingest_session_content(prompt, reply)
                return_code = 0
            except httpx.TimeoutException:
                _write_qwenchat_status(
                    session_id=session_id,
                    model=str(args.model),
                    status='timeout',
                    phase='model_request',
                    message=f'request timed out after {float(args.timeout_sec):.0f}s',
                    mode=mode,
                    base_url=base_url,
                )
                print(f'[qwenchat] request timed out after {float(args.timeout_sec):.0f}s', file=sys.stderr)
            except httpx.HTTPError as exc:
                _write_qwenchat_status(
                    session_id=session_id,
                    model=str(args.model),
                    status='error',
                    phase='model_request',
                    message=str(exc),
                    mode=mode,
                    base_url=base_url,
                )
                print(f'[qwenchat] request failed: {exc}', file=sys.stderr)
        else:
            _write_qwenchat_status(
                session_id=session_id,
                model=str(args.model),
                status='running',
                phase='interactive',
                message='interactive chat loop active',
                mode=mode,
                base_url=base_url,
            )
            return_code = _interactive_loop(
                base_url=base_url,
                model=str(args.model),
                timeout_sec=float(args.timeout_sec),
                session_id=session_id,
            )
    finally:
        stop_event.set()
        heartbeat.join(timeout=1.0)
        coordinator.clear_owner(owner, metadata={'exclusive': False, 'exclusive_owner': '', 'task': 'interactive_chat'})
        append_session_event(
            interface='qwenchat',
            session_id=session_id,
            event_type='end',
            summary=f'qwenchat session ended with code {return_code}',
            metadata={'model': str(args.model)},
        )
        _write_qwenchat_status(
            session_id=session_id,
            model=str(args.model),
            status='success' if return_code == 0 else 'error',
            phase='completed',
            message=f'qwenchat session ended with code {return_code}',
            mode=mode,
            return_code=return_code,
            base_url=base_url,
        )

    return int(return_code)


if __name__ == '__main__':
    raise SystemExit(main())
