"""
Event system for Stratum simulation.

This module provides a structured event API that allows external systems
(games, visualizations, analytics) to subscribe to simulation events
without polling raw fields. Events are emitted during simulation and
stored in a ring buffer for efficient access.

Key Features:
- Structured event types with rich metadata
- Ring buffer storage with configurable size
- Subscription/callback system for event handlers
- Event filtering by type and region
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum, auto
from collections import deque
import time


class EventType(Enum):
    """Types of events emitted by the simulation."""
    
    # Species and material events
    SPECIES_CREATED = auto()
    SPECIES_DESTROYED = auto()
    
    # High-energy physics events
    FUSION_OCCURRED = auto()
    DECAY_OCCURRED = auto()
    DEGENERATE_TRANSITION = auto()
    BLACK_HOLE_FORMED = auto()
    BLACK_HOLE_ABSORPTION = auto()
    
    # Thermodynamic events
    SHOCK_HEATING = auto()
    RADIATION_EMISSION = auto()
    
    # Transport events
    MASS_TRANSFER = auto()
    BOUNDARY_OUTFLOW = auto()
    BOUNDARY_INFLOW = auto()
    
    # Signal events
    SIGNAL_EMITTED = auto()
    SIGNAL_DELIVERED = auto()
    
    # Regime transitions
    REGIME_TRANSITION = auto()  # Cell crossed a Z threshold
    
    # Diagnostic events
    CONSERVATION_VIOLATION = auto()
    NUMERICAL_INSTABILITY = auto()
    
    # Tick lifecycle events
    TICK_STARTED = auto()
    TICK_COMPLETED = auto()


@dataclass
class SimulationEvent:
    """A structured event emitted during simulation.
    
    Attributes:
        event_type: Type of event.
        tick: Simulation tick when event occurred.
        timestamp: Wall-clock time when event was created.
        cell: Grid cell (i, j) where event occurred, or None for global events.
        data: Event-specific payload data.
        priority: Event priority (higher = more important).
    """
    event_type: EventType
    tick: int
    timestamp: float
    cell: Optional[Tuple[int, int]]
    data: Dict[str, Any]
    priority: int = 0

    def __repr__(self) -> str:
        cell_str = f"@{self.cell}" if self.cell else "@global"
        return f"Event({self.event_type.name}, tick={self.tick}, {cell_str})"


# Type alias for event handlers
EventHandler = Callable[[SimulationEvent], None]


class EventBus:
    """Event bus for simulation event distribution.
    
    The EventBus provides a publish-subscribe interface for simulation events.
    Events are stored in a ring buffer and can be retrieved by subscribers
    either via callbacks or by polling.
    
    Example:
        bus = EventBus(buffer_size=1000)
        bus.subscribe(EventType.BLACK_HOLE_FORMED, on_bh_formed)
        bus.emit(EventType.BLACK_HOLE_FORMED, tick=100, cell=(5, 5), data={'mass': 10.0})
        events = bus.get_events(EventType.BLACK_HOLE_FORMED)
    """

    def __init__(self, buffer_size: int = 10000):
        """Initialize the event bus.
        
        Args:
            buffer_size: Maximum number of events to retain in the ring buffer.
        """
        self.buffer_size = buffer_size
        self._events: deque[SimulationEvent] = deque(maxlen=buffer_size)
        self._handlers: Dict[EventType, List[EventHandler]] = {}
        self._global_handlers: List[EventHandler] = []
        self._type_counts: Dict[EventType, int] = {t: 0 for t in EventType}
        self._enabled: bool = True

    def emit(
        self,
        event_type: EventType,
        tick: int,
        cell: Optional[Tuple[int, int]] = None,
        data: Optional[Dict[str, Any]] = None,
        priority: int = 0
    ) -> Optional[SimulationEvent]:
        """Emit a new event.
        
        Args:
            event_type: Type of event.
            tick: Current simulation tick.
            cell: Grid cell where event occurred, or None for global.
            data: Event-specific payload data.
            priority: Event priority.
            
        Returns:
            The created event, or None if event bus is disabled.
        """
        if not self._enabled:
            return None
        
        event = SimulationEvent(
            event_type=event_type,
            tick=tick,
            timestamp=time.time(),
            cell=cell,
            data=data or {},
            priority=priority
        )
        
        # Add to ring buffer
        self._events.append(event)
        self._type_counts[event_type] += 1
        
        # Dispatch to handlers
        self._dispatch(event)
        
        return event

    def subscribe(
        self,
        event_type: Optional[EventType],
        handler: EventHandler
    ) -> None:
        """Subscribe to events of a specific type.
        
        Args:
            event_type: Type to subscribe to, or None for all events.
            handler: Callback function to invoke when event occurs.
        """
        if event_type is None:
            self._global_handlers.append(handler)
        else:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(handler)

    def unsubscribe(
        self,
        event_type: Optional[EventType],
        handler: EventHandler
    ) -> bool:
        """Unsubscribe a handler from events.
        
        Args:
            event_type: Type to unsubscribe from, or None for global.
            handler: Handler to remove.
            
        Returns:
            True if handler was found and removed.
        """
        if event_type is None:
            if handler in self._global_handlers:
                self._global_handlers.remove(handler)
                return True
        else:
            if event_type in self._handlers and handler in self._handlers[event_type]:
                self._handlers[event_type].remove(handler)
                return True
        return False

    def get_events(
        self,
        event_type: Optional[EventType] = None,
        since_tick: Optional[int] = None,
        cell: Optional[Tuple[int, int]] = None,
        limit: Optional[int] = None
    ) -> List[SimulationEvent]:
        """Retrieve events matching the specified filters.
        
        Args:
            event_type: Filter by event type, or None for all types.
            since_tick: Only return events from this tick onwards.
            cell: Filter by cell location.
            limit: Maximum number of events to return.
            
        Returns:
            List of matching events (newest first).
        """
        result = []
        
        for event in reversed(self._events):
            # Apply filters
            if event_type is not None and event.event_type != event_type:
                continue
            if since_tick is not None and event.tick < since_tick:
                continue
            if cell is not None and event.cell != cell:
                continue
            
            result.append(event)
            
            if limit is not None and len(result) >= limit:
                break
        
        return result

    def get_events_for_tick(self, tick: int) -> List[SimulationEvent]:
        """Get all events for a specific tick.
        
        Args:
            tick: Tick number.
            
        Returns:
            List of events from that tick.
        """
        return [e for e in self._events if e.tick == tick]

    def get_event_counts(self) -> Dict[EventType, int]:
        """Get counts of events by type.
        
        Returns:
            Dictionary mapping event types to counts.
        """
        return dict(self._type_counts)

    def clear(self) -> None:
        """Clear all stored events."""
        self._events.clear()
        self._type_counts = {t: 0 for t in EventType}

    def enable(self) -> None:
        """Enable event emission."""
        self._enabled = True

    def disable(self) -> None:
        """Disable event emission (events are dropped)."""
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        """Check if event bus is enabled."""
        return self._enabled

    @property
    def event_count(self) -> int:
        """Get total number of events in buffer."""
        return len(self._events)

    def _dispatch(self, event: SimulationEvent) -> None:
        """Dispatch event to registered handlers.
        
        Args:
            event: Event to dispatch.
        """
        # Type-specific handlers
        if event.event_type in self._handlers:
            for handler in self._handlers[event.event_type]:
                try:
                    handler(event)
                except Exception:
                    pass  # Don't let handler errors break simulation
        
        # Global handlers
        for handler in self._global_handlers:
            try:
                handler(event)
            except Exception:
                pass


# Event helper functions for common events

def emit_species_created(
    bus: EventBus,
    tick: int,
    species_id: str,
    he_props: Dict[str, float],
    provenance: Optional[Dict[str, Any]] = None
) -> None:
    """Emit a species created event.
    
    Args:
        bus: Event bus instance.
        tick: Current tick.
        species_id: ID of the created species.
        he_props: High-energy properties of the species.
        provenance: Provenance information.
    """
    bus.emit(
        EventType.SPECIES_CREATED,
        tick=tick,
        data={
            'species_id': species_id,
            'he_props': he_props,
            'provenance': provenance or {}
        }
    )


def emit_black_hole_formed(
    bus: EventBus,
    tick: int,
    cell: Tuple[int, int],
    initial_mass: float,
    trigger_Z: float
) -> None:
    """Emit a black hole formation event.
    
    Args:
        bus: Event bus instance.
        tick: Current tick.
        cell: Cell where BH formed.
        initial_mass: Initial BH mass.
        trigger_Z: Z value that triggered formation.
    """
    bus.emit(
        EventType.BLACK_HOLE_FORMED,
        tick=tick,
        cell=cell,
        data={
            'initial_mass': initial_mass,
            'trigger_Z': trigger_Z
        },
        priority=10  # High priority event
    )


def emit_fusion_occurred(
    bus: EventBus,
    tick: int,
    cell: Tuple[int, int],
    parent_species: str,
    child_species: str,
    mass_converted: float,
    energy_released: float
) -> None:
    """Emit a fusion event.
    
    Args:
        bus: Event bus instance.
        tick: Current tick.
        cell: Cell where fusion occurred.
        parent_species: Parent species ID.
        child_species: Child species ID.
        mass_converted: Amount of mass that underwent fusion.
        energy_released: Energy released (or consumed if negative).
    """
    bus.emit(
        EventType.FUSION_OCCURRED,
        tick=tick,
        cell=cell,
        data={
            'parent_species': parent_species,
            'child_species': child_species,
            'mass_converted': mass_converted,
            'energy_released': energy_released
        }
    )


def emit_regime_transition(
    bus: EventBus,
    tick: int,
    cell: Tuple[int, int],
    from_regime: str,
    to_regime: str,
    Z_value: float
) -> None:
    """Emit a regime transition event.
    
    Args:
        bus: Event bus instance.
        tick: Current tick.
        cell: Cell where transition occurred.
        from_regime: Previous regime name.
        to_regime: New regime name.
        Z_value: Current Z value.
    """
    bus.emit(
        EventType.REGIME_TRANSITION,
        tick=tick,
        cell=cell,
        data={
            'from_regime': from_regime,
            'to_regime': to_regime,
            'Z_value': Z_value
        }
    )


def emit_tick_completed(
    bus: EventBus,
    tick: int,
    total_mass: float,
    total_energy: float,
    active_cells: int,
    microticks_processed: int
) -> None:
    """Emit a tick completed event.
    
    Args:
        bus: Event bus instance.
        tick: Completed tick number.
        total_mass: Total mass in simulation.
        total_energy: Total energy in simulation.
        active_cells: Number of active cells processed.
        microticks_processed: Total microticks processed this tick.
    """
    bus.emit(
        EventType.TICK_COMPLETED,
        tick=tick,
        data={
            'total_mass': total_mass,
            'total_energy': total_energy,
            'active_cells': active_cells,
            'microticks_processed': microticks_processed
        }
    )
