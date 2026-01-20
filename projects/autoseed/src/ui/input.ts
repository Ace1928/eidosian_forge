import type { GameState, Vector2 } from "../core/types.js";
import { orbitPosition } from "../core/animation.js";
import { listSystems } from "../core/procgen.js";
import type { ViewState } from "./render.js";

const distance = (a: Vector2, b: Vector2): number =>
  Math.hypot(a.x - b.x, a.y - b.y);

export interface InputState {
  keys: Set<string>;
  dragging: boolean;
  lastPointer: Vector2 | null;
  dragMoved: boolean;
}

const toWorld = (canvas: HTMLCanvasElement, view: ViewState, screen: Vector2): Vector2 => {
  const rect = canvas.getBoundingClientRect();
  const localX = screen.x - rect.left;
  const localY = screen.y - rect.top;
  return {
    x: (localX - rect.width / 2) / view.zoom + view.camera.x,
    y: (localY - rect.height / 2) / view.zoom + view.camera.y
  };
};

export const attachInput = (
  canvas: HTMLCanvasElement,
  getState: () => GameState,
  view: ViewState,
  getTime: () => number
): InputState => {
  const input: InputState = {
    keys: new Set<string>(),
    dragging: false,
    lastPointer: null,
    dragMoved: false
  };

  canvas.addEventListener("click", (event) => {
    if (input.dragMoved) {
      input.dragMoved = false;
      return;
    }
    const click = toWorld(canvas, view, { x: event.clientX, y: event.clientY });
    const systems = listSystems(getState().galaxy);
    let selectedBody: string | null = null;
    let selectedSystem = view.selectedSystemId;
    let bestBodyDistance = Number.POSITIVE_INFINITY;
    const time = getTime();

    for (const system of systems) {
      const systemDistance = distance(click, system.position);
      if (systemDistance < 80) {
        const local = {
          x: click.x - system.position.x,
          y: click.y - system.position.y
        };
        for (const body of system.bodies) {
          const pos = orbitPosition(body, time, { x: 0, y: 0 });
          const bodyDistance = distance(local, pos);
          if (bodyDistance < 10 && bodyDistance < bestBodyDistance) {
            selectedBody = body.id;
            selectedSystem = system.id;
            bestBodyDistance = bodyDistance;
          }
        }
        if (!selectedBody) {
          selectedSystem = system.id;
        }
      }
    }

    view.selectedSystemId = selectedSystem;
    view.selectedBodyId = selectedBody;
  });

  window.addEventListener("keydown", (event) => {
    const key = event.key.toLowerCase();
    if (key === " ") {
      view.paused = !view.paused;
    }
    if (key === "+" || key === "=" || event.code === "NumpadAdd") {
      view.speed = Math.min(4, view.speed + 0.5);
    }
    if (key === "-" || key === "_" || event.code === "NumpadSubtract") {
      view.speed = Math.max(0.5, view.speed - 0.5);
    }
    if (key === "z") {
      view.zoom = Math.max(0.4, view.zoom - 0.1);
    }
    if (key === "x") {
      view.zoom = Math.min(2.4, view.zoom + 0.1);
    }
    if (key === "0") {
      view.zoom = 1;
    }
    if (["w", "a", "s", "d", "arrowup", "arrowdown", "arrowleft", "arrowright"].includes(key)) {
      input.keys.add(key);
    }
  });

  window.addEventListener("keyup", (event) => {
    input.keys.delete(event.key.toLowerCase());
  });

  const startDrag = (x: number, y: number): void => {
    input.dragging = true;
    input.lastPointer = { x, y };
    input.dragMoved = false;
  };

  const moveDrag = (x: number, y: number): void => {
    if (!input.dragging || !input.lastPointer) {
      return;
    }
    const delta = { x: x - input.lastPointer.x, y: y - input.lastPointer.y };
    if (Math.hypot(delta.x, delta.y) > 2) {
      input.dragMoved = true;
    }
    view.camera = {
      x: view.camera.x - delta.x / view.zoom,
      y: view.camera.y - delta.y / view.zoom
    };
    input.lastPointer = { x, y };
  };

  const endDrag = (): void => {
    input.dragging = false;
    input.lastPointer = null;
  };

  canvas.addEventListener("mousedown", (event) => startDrag(event.clientX, event.clientY));
  window.addEventListener("mousemove", (event) => moveDrag(event.clientX, event.clientY));
  window.addEventListener("mouseup", endDrag);

  canvas.addEventListener("touchstart", (event) => {
    const touch = event.touches[0];
    if (!touch) {
      return;
    }
    startDrag(touch.clientX, touch.clientY);
  });
  canvas.addEventListener("touchmove", (event) => {
    const touch = event.touches[0];
    if (!touch) {
      return;
    }
    moveDrag(touch.clientX, touch.clientY);
  });
  canvas.addEventListener("touchend", endDrag);

  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    const zoomDelta = event.deltaY > 0 ? -0.08 : 0.08;
    view.zoom = Math.min(2.6, Math.max(0.4, view.zoom + zoomDelta));
  }, { passive: false });

  return input;
};
