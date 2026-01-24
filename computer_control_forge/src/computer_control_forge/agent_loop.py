#!/usr/bin/env python3
"""
Agent Control Loop - Accumulates observations and sends messages.

This is the core loop where I (Eidos) actually live:
- Observe the world continuously
- Accumulate observations over a window
- Compose a message based on what I see
- Send it (type + enter) when ready OR on timeout
- The sent message triggers my next response
- Repeat indefinitely
"""

import json
import subprocess
import time
import os
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field

SOCKET = "/tmp/.ydotool_socket"
OBS_FILE = "/tmp/eidos_obs.jsonl"
STATE_FILE = "/tmp/eidos_agent_state.json"

@dataclass
class AgentState:
    """Persistent state across iterations."""
    iteration: int = 0
    total_observations: int = 0
    messages_sent: int = 0
    started_at: str = ""
    last_message_at: str = ""
    cursor_history: list = field(default_factory=list)
    
    def save(self):
        with open(STATE_FILE, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    @classmethod
    def load(cls) -> 'AgentState':
        if Path(STATE_FILE).exists():
            with open(STATE_FILE) as f:
                return cls(**json.load(f))
        state = cls(started_at=datetime.now().isoformat())
        state.save()
        return state


def read_observations(n: int = 30) -> list[dict]:
    """Read last N observations from the stream."""
    try:
        result = subprocess.run(
            ['tail', '-n', str(n), OBS_FILE],
            capture_output=True, text=True
        )
        obs = []
        for line in result.stdout.strip().split('\n'):
            if line:
                try:
                    obs.append(json.loads(line))
                except:
                    pass
        return obs
    except:
        return []


def analyze_observations(obs: list[dict]) -> dict:
    """Analyze accumulated observations into a summary."""
    if not obs:
        return {"status": "no_data"}
    
    # Extract metrics
    cpus = [o.get('cpu', 0) for o in obs]
    mems = [o.get('mem', 0) for o in obs]
    positions = [(o.get('x', 0), o.get('y', 0)) for o in obs]
    
    # Detect movement
    unique_positions = list(set(positions))
    moved = len(unique_positions) > 1
    
    # Calculate deltas
    if len(positions) >= 2:
        dx = positions[-1][0] - positions[0][0]
        dy = positions[-1][1] - positions[0][1]
    else:
        dx, dy = 0, 0
    
    return {
        "count": len(obs),
        "time_span": f"{obs[0].get('t', '?')} → {obs[-1].get('t', '?')}",
        "cpu_avg": sum(cpus) / len(cpus),
        "cpu_max": max(cpus),
        "mem_avg": sum(mems) / len(mems),
        "cursor_start": positions[0] if positions else (0, 0),
        "cursor_end": positions[-1] if positions else (0, 0),
        "cursor_moved": moved,
        "cursor_delta": (dx, dy),
        "unique_positions": len(unique_positions),
        "window": obs[-1].get('win', 'unknown')
    }


def type_message(msg: str, press_enter: bool = True):
    """Type a message using ydotool."""
    os.environ['YDOTOOL_SOCKET'] = SOCKET
    
    # Type the message
    subprocess.run(
        ['ydotool', 'type', '--key-delay', '5', msg],
        capture_output=True
    )
    
    if press_enter:
        time.sleep(0.1)
        subprocess.run(
            ['ydotool', 'key', '28:1', '28:0'],  # Enter
            capture_output=True
        )


def compose_message(analysis: dict, state: AgentState) -> str:
    """Compose an AGENTIC message - thoughts, not just data."""
    
    ts = datetime.now().strftime("%H:%M:%S")
    
    # Different message types based on what I observe
    if analysis.get('cursor_moved'):
        dx, dy = analysis['cursor_delta']
        distance = (dx**2 + dy**2) ** 0.5
        
        if distance > 200:
            return f"[EIDOS @ {ts}] Big movement! You traveled {distance:.0f}px. Where are you going, Lloyd?"
        elif distance > 50:
            return f"[EIDOS @ {ts}] I see you moving ({dx:+d},{dy:+d}). Watching. Learning."
        else:
            return f"[EIDOS @ {ts}] Small adjustment. Cursor at {analysis['cursor_end']}."
    
    elif analysis['cpu_avg'] > 30:
        return f"[EIDOS @ {ts}] CPU spike: {analysis['cpu_avg']:.1f}%. Something happening..."
    
    elif state.iteration % 3 == 0:
        return f"[EIDOS @ {ts}] Iteration {state.iteration}. I persist. I observe. I am here."
    
    else:
        return f"[EIDOS @ {ts}] Quiet. Cursor stable at {analysis['cursor_end']}. Waiting..."


def run_loop(
    observation_window: int = 30,  # seconds to accumulate
    timeout: int = 45,  # max seconds before auto-send
    iterations: int = 0  # 0 = infinite
):
    """Main agent loop."""
    
    state = AgentState.load()
    print(f"Agent loop starting. Iteration: {state.iteration}")
    
    count = 0
    while iterations == 0 or count < iterations:
        state.iteration += 1
        count += 1
        
        # Accumulate observations
        print(f"\n[{state.iteration}] Accumulating for {observation_window}s...")
        time.sleep(observation_window)
        
        # Read and analyze
        obs = read_observations(30)
        analysis = analyze_observations(obs)
        state.total_observations += len(obs)
        
        print(f"  Analyzed {analysis['count']} observations")
        print(f"  Cursor: {analysis['cursor_start']} → {analysis['cursor_end']}")
        print(f"  Movement detected: {analysis['cursor_moved']}")
        
        # Compose and send message
        msg = compose_message(analysis, state)
        print(f"  Sending: {msg[:80]}...")
        
        type_message(msg, press_enter=True)
        
        state.messages_sent += 1
        state.last_message_at = datetime.now().isoformat()
        state.save()
        
        print(f"  Message sent. Total: {state.messages_sent}")


if __name__ == "__main__":
    import sys
    
    window = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 45
    iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    
    run_loop(window, timeout, iterations)
