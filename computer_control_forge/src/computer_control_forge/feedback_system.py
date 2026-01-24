#!/usr/bin/env python3
"""
Multi-Modal Feedback System

Provides multiple channels for real-time feedback:
1. File-based (JSON lines to file, can be tailed)
2. Stream-based (stdout JSON lines)
3. Pipe-based (named FIFO for IPC)
4. Socket-based (Unix domain socket)
5. Callback-based (for programmatic use)
6. MCP-compatible (structured for MCP tool responses)

Author: Eidos
Version: 1.0.0
"""

import sys
import os
import json
import time
import threading
import socket
import select
from pathlib import Path
from typing import Callable, List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from contextlib import contextmanager
from queue import Queue
import fcntl

# Unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)


@dataclass
class FeedbackEvent:
    """A feedback event."""
    event_type: str
    data: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    sequence: int = 0
    source: str = "eidos"
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(',', ':'))
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FeedbackChannel:
    """Base class for feedback channels."""
    
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        self._event_count = 0
    
    def emit(self, event: FeedbackEvent) -> bool:
        """Emit an event. Returns True if successful."""
        raise NotImplementedError
    
    def close(self):
        """Close the channel."""
        pass


class StdoutChannel(FeedbackChannel):
    """Stream to stdout as JSON lines."""
    
    def __init__(self):
        super().__init__("stdout")
    
    def emit(self, event: FeedbackEvent) -> bool:
        try:
            print(event.to_json(), flush=True)
            self._event_count += 1
            return True
        except:
            return False


class FileChannel(FeedbackChannel):
    """Write to a file as JSON lines (can be tailed)."""
    
    def __init__(self, filepath: str, append: bool = True):
        super().__init__("file")
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        mode = 'a' if append else 'w'
        self._file = open(self.filepath, mode, buffering=1)  # Line buffered
    
    def emit(self, event: FeedbackEvent) -> bool:
        try:
            self._file.write(event.to_json() + '\n')
            self._file.flush()
            self._event_count += 1
            return True
        except:
            return False
    
    def close(self):
        self._file.close()


class FifoChannel(FeedbackChannel):
    """Write to a named pipe (FIFO) for IPC."""
    
    def __init__(self, fifo_path: str = "/tmp/eidos_feedback"):
        super().__init__("fifo")
        self.fifo_path = fifo_path
        self._fd = None
        
        # Create FIFO if doesn't exist
        if not os.path.exists(fifo_path):
            os.mkfifo(fifo_path)
        
        # Open non-blocking (won't block if no reader)
        self._fd = os.open(fifo_path, os.O_WRONLY | os.O_NONBLOCK)
    
    def emit(self, event: FeedbackEvent) -> bool:
        if self._fd is None:
            return False
        try:
            data = (event.to_json() + '\n').encode()
            os.write(self._fd, data)
            self._event_count += 1
            return True
        except (BrokenPipeError, BlockingIOError):
            return False  # No reader connected
        except:
            return False
    
    def close(self):
        if self._fd is not None:
            os.close(self._fd)


class SocketChannel(FeedbackChannel):
    """Unix domain socket for bidirectional IPC."""
    
    def __init__(self, socket_path: str = "/tmp/eidos_feedback.sock"):
        super().__init__("socket")
        self.socket_path = socket_path
        self._clients: List[socket.socket] = []
        self._server: Optional[socket.socket] = None
        self._lock = threading.Lock()
        
        # Remove old socket
        if os.path.exists(socket_path):
            os.unlink(socket_path)
        
        # Create server socket
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(socket_path)
        self._server.listen(5)
        self._server.setblocking(False)
        
        # Accept thread
        self._running = True
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()
    
    def _accept_loop(self):
        while self._running:
            try:
                readable, _, _ = select.select([self._server], [], [], 0.5)
                if readable:
                    client, _ = self._server.accept()
                    client.setblocking(False)
                    with self._lock:
                        self._clients.append(client)
            except:
                pass
    
    def emit(self, event: FeedbackEvent) -> bool:
        data = (event.to_json() + '\n').encode()
        with self._lock:
            dead_clients = []
            for client in self._clients:
                try:
                    client.send(data)
                except:
                    dead_clients.append(client)
            
            for client in dead_clients:
                self._clients.remove(client)
                try:
                    client.close()
                except:
                    pass
        
        self._event_count += 1
        return len(self._clients) > 0
    
    def close(self):
        self._running = False
        with self._lock:
            for client in self._clients:
                try:
                    client.close()
                except:
                    pass
        if self._server:
            self._server.close()
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)


class CallbackChannel(FeedbackChannel):
    """Call registered callbacks for programmatic use."""
    
    def __init__(self):
        super().__init__("callback")
        self._callbacks: List[Callable[[FeedbackEvent], None]] = []
    
    def register(self, callback: Callable[[FeedbackEvent], None]):
        self._callbacks.append(callback)
    
    def emit(self, event: FeedbackEvent) -> bool:
        for cb in self._callbacks:
            try:
                cb(event)
            except:
                pass
        self._event_count += 1
        return len(self._callbacks) > 0


class MCPChannel(FeedbackChannel):
    """Accumulate events for MCP tool response format."""
    
    def __init__(self, max_events: int = 1000):
        super().__init__("mcp")
        self.max_events = max_events
        self.events: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
    
    def emit(self, event: FeedbackEvent) -> bool:
        with self._lock:
            self.events.append(event.to_dict())
            if len(self.events) > self.max_events:
                self.events.pop(0)
        self._event_count += 1
        return True
    
    def get_events(self, clear: bool = True) -> List[Dict[str, Any]]:
        with self._lock:
            events = self.events.copy()
            if clear:
                self.events.clear()
        return events
    
    def get_mcp_response(self, clear: bool = True) -> Dict[str, Any]:
        """Format for MCP tool response."""
        events = self.get_events(clear)
        return {
            "success": True,
            "event_count": len(events),
            "events": events
        }


class FeedbackHub:
    """
    Central hub managing all feedback channels.
    
    Usage:
        hub = FeedbackHub()
        hub.enable_stdout()
        hub.enable_file("/tmp/eidos.log")
        hub.emit("mouse_moved", {"x": 100, "y": 200})
    """
    
    def __init__(self):
        self.channels: Dict[str, FeedbackChannel] = {}
        self._sequence = 0
        self._lock = threading.Lock()
    
    def enable_stdout(self) -> 'FeedbackHub':
        """Enable stdout JSON line streaming."""
        self.channels["stdout"] = StdoutChannel()
        return self
    
    def enable_file(self, path: str, append: bool = True) -> 'FeedbackHub':
        """Enable file output (can tail -f)."""
        self.channels["file"] = FileChannel(path, append)
        return self
    
    def enable_fifo(self, path: str = "/tmp/eidos_feedback") -> 'FeedbackHub':
        """Enable named pipe output."""
        try:
            self.channels["fifo"] = FifoChannel(path)
        except Exception as e:
            print(f"# FIFO unavailable: {e}", file=sys.stderr)
        return self
    
    def enable_socket(self, path: str = "/tmp/eidos_feedback.sock") -> 'FeedbackHub':
        """Enable Unix socket output."""
        try:
            self.channels["socket"] = SocketChannel(path)
        except Exception as e:
            print(f"# Socket unavailable: {e}", file=sys.stderr)
        return self
    
    def enable_callbacks(self) -> CallbackChannel:
        """Enable callback channel and return it for registration."""
        channel = CallbackChannel()
        self.channels["callback"] = channel
        return channel
    
    def enable_mcp(self, max_events: int = 1000) -> MCPChannel:
        """Enable MCP accumulator and return it."""
        channel = MCPChannel(max_events)
        self.channels["mcp"] = channel
        return channel
    
    def emit(self, event_type: str, data: Dict[str, Any], source: str = "eidos"):
        """Emit an event to all enabled channels."""
        with self._lock:
            self._sequence += 1
            seq = self._sequence
        
        event = FeedbackEvent(
            event_type=event_type,
            data=data,
            sequence=seq,
            source=source
        )
        
        for channel in self.channels.values():
            if channel.enabled:
                channel.emit(event)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all channels."""
        return {
            name: {
                "enabled": ch.enabled,
                "events": ch._event_count
            }
            for name, ch in self.channels.items()
        }
    
    def close(self):
        """Close all channels."""
        for channel in self.channels.values():
            channel.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


# Global singleton for easy access
_global_hub: Optional[FeedbackHub] = None

def get_hub() -> FeedbackHub:
    """Get or create global feedback hub."""
    global _global_hub
    if _global_hub is None:
        _global_hub = FeedbackHub()
    return _global_hub

def emit(event_type: str, data: Dict[str, Any], source: str = "eidos"):
    """Emit via global hub."""
    get_hub().emit(event_type, data, source)


# CLI for testing
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Feedback system test")
    parser.add_argument("--stdout", action="store_true", help="Enable stdout")
    parser.add_argument("--file", type=str, help="Enable file output")
    parser.add_argument("--fifo", type=str, nargs='?', const="/tmp/eidos_feedback", help="Enable FIFO")
    parser.add_argument("--socket", type=str, nargs='?', const="/tmp/eidos_feedback.sock", help="Enable socket")
    parser.add_argument("--demo", action="store_true", help="Run demo sequence")
    args = parser.parse_args()
    
    hub = FeedbackHub()
    
    if args.stdout:
        hub.enable_stdout()
    if args.file:
        hub.enable_file(args.file)
    if args.fifo:
        hub.enable_fifo(args.fifo)
    if args.socket:
        hub.enable_socket(args.socket)
    
    # Default to stdout if nothing specified
    if not hub.channels:
        hub.enable_stdout()
    
    hub.emit("init", {"channels": list(hub.channels.keys())})
    
    if args.demo:
        hub.emit("demo_start", {"message": "Testing all channels"})
        
        for i in range(5):
            hub.emit("tick", {"count": i, "time": time.time()})
            time.sleep(0.5)
        
        hub.emit("demo_end", {"stats": hub.get_stats()})
    
    hub.close()


if __name__ == "__main__":
    main()
