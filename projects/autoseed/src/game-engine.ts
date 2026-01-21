import { advanceTick, createInitialState, queueStructure } from "./core/simulation.js";
import { ensureSystem, getSystemSpacing, listSystems } from "./core/procgen.js";
import { orbitPosition } from "./core/animation.js";
import type { GameCommand } from "./core/commands.js";
import type { GameConfig, GameState, Vector2 } from "./core/types.js";
import { CommandQueue } from "./core/command-queue.js";
import { resolveCanvasMetrics } from "./core/canvas.js";
import { buildBodyIndex } from "./core/selectors.js";
import { attachInput, type InputBinding, type InputState } from "./ui/input.js";
import { renderFrame, type ViewState } from "./ui/render.js";
import {
  commandFromBuildRequest,
  commandFromPanelAction,
  commandFromProbeDesign
} from "./ui/panel-commands.js";
import { updatePanels } from "./ui/panels.js";

const clamp = (value: number, min: number, max: number): number =>
  Math.max(min, Math.min(max, value));

const distance = (a: Vector2, b: Vector2): number => Math.hypot(a.x - b.x, a.y - b.y);

export class GameEngine {
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly input: InputState;
  private readonly inputBinding: InputBinding;
  private readonly tickDuration = 1500;
  private state: GameState;
  private view: ViewState;
  private viewTime = 0;
  private last = 0;
  private accumulator = 0;
  private frameSize = { width: 0, height: 0 };
  private commandQueue = new CommandQueue();
  private frameRequest: number | null = null;
  private profiling = {
    enabled: false,
    frames: 0,
    simMs: 0,
    renderMs: 0,
    frameMs: 0,
    lastLog: 0
  };

  constructor(canvas: HTMLCanvasElement, config: GameConfig) {
    this.canvas = canvas;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      throw new Error("Canvas context not available");
    }
    this.ctx = ctx;

    this.state = createInitialState(config);
    this.view = {
      selectedSystemId: listSystems(this.state.galaxy)[0]?.id ?? "",
      selectedBodyId: null,
      paused: false,
      speed: 0.6,
      camera: { x: 0, y: 0 },
      zoom: 1
    };

    this.resize();
    window.addEventListener("resize", this.resize);

    this.inputBinding = attachInput(this.canvas, this.enqueue);
    this.input = this.inputBinding.state;
    this.profiling.enabled = new URLSearchParams(window.location.search).has("profile");
  }

  start(): void {
    if (this.frameRequest !== null) {
      return;
    }
    this.last = performance.now();
    this.profiling.lastLog = this.last;
    this.frameRequest = requestAnimationFrame(this.loop);
  }

  destroy(): void {
    if (this.frameRequest !== null) {
      window.cancelAnimationFrame(this.frameRequest);
      this.frameRequest = null;
    }
    window.removeEventListener("resize", this.resize);
    this.inputBinding.destroy();
  }

  private enqueue = (command: GameCommand): void => {
    this.commandQueue.enqueue(command);
  };

  private resize = (): void => {
    const metrics = resolveCanvasMetrics(
      window.innerWidth,
      window.innerHeight,
      window.devicePixelRatio ?? 1
    );
    this.canvas.style.width = `${metrics.cssWidth}px`;
    this.canvas.style.height = `${metrics.cssHeight}px`;
    this.canvas.width = metrics.pixelWidth;
    this.canvas.height = metrics.pixelHeight;
    this.ctx.setTransform(metrics.ratio, 0, 0, metrics.ratio, 0, 0);
    this.frameSize = { width: metrics.cssWidth, height: metrics.cssHeight };
  };

  private loop = (time: number): void => {
    if (this.frameRequest === null) {
      return;
    }
    const delta = time - this.last;
    this.last = time;
    const frameStart = performance.now();

    this.processCommands();
    this.applyCameraKeys(delta);
    this.ensureSystemsInView();
    if (this.state.outcome && !this.view.paused) {
      this.view.paused = true;
    }

    if (!this.view.paused) {
      this.accumulator += delta * this.view.speed;
      this.viewTime += delta * this.view.speed;
    }
    while (this.accumulator >= this.tickDuration) {
      this.state = advanceTick(this.state);
      this.accumulator -= this.tickDuration;
    }
    const simEnd = performance.now();

    renderFrame(this.ctx, this.state, this.view, this.viewTime, this.frameSize);
    updatePanels(
      this.state,
      this.view,
      (payload) => this.enqueue(commandFromBuildRequest(payload)),
      (action) => this.enqueue(commandFromPanelAction(action)),
      (design) => this.enqueue(commandFromProbeDesign(design))
    );
    const frameEnd = performance.now();
    this.recordProfile(frameEnd - frameStart, simEnd - frameStart, frameEnd - simEnd, frameEnd);
    this.frameRequest = requestAnimationFrame(this.loop);
  };

  private applyCameraKeys(delta: number): void {
    const speed = 0.25 * delta;
    if (this.input.keys.has("w") || this.input.keys.has("arrowup")) {
      this.view.camera.y -= speed;
    }
    if (this.input.keys.has("s") || this.input.keys.has("arrowdown")) {
      this.view.camera.y += speed;
    }
    if (this.input.keys.has("a") || this.input.keys.has("arrowleft")) {
      this.view.camera.x -= speed;
    }
    if (this.input.keys.has("d") || this.input.keys.has("arrowright")) {
      this.view.camera.x += speed;
    }
  }

  private processCommands(): void {
    if (this.commandQueue.size() === 0) {
      return;
    }
    const commands = this.commandQueue.drain();
    for (const command of commands) {
      this.applyCommand(command);
    }
  }

  private applyCommand(command: GameCommand): void {
    switch (command.type) {
      case "pan-camera":
        this.view.camera = {
          x: this.view.camera.x - command.delta.x / this.view.zoom,
          y: this.view.camera.y - command.delta.y / this.view.zoom
        };
        return;
      case "select-at":
        this.selectAt(command.screen);
        return;
      case "toggle-pause":
        this.view.paused = !this.view.paused;
        return;
      case "speed-change":
        this.view.speed = clamp(this.view.speed + command.delta, 0.5, 4);
        return;
      case "zoom-change":
        this.view.zoom = clamp(this.view.zoom + command.delta, 0.4, 2.6);
        return;
      case "zoom-set":
        this.view.zoom = clamp(command.value, 0.4, 2.6);
        return;
      case "center-selected":
        if (this.view.selectedBodyId) {
          const info = this.getBodyWorldPosition(this.view.selectedBodyId);
          if (info) {
            this.view.camera = { ...info.position };
            this.view.selectedSystemId = info.systemId;
          }
        }
        return;
      case "center-probe": {
        const player = this.state.factions[0];
        const probe = player?.probes[0];
        if (probe) {
          const info = this.getBodyWorldPosition(probe.bodyId);
          if (info) {
            this.view.camera = { ...info.position };
            this.view.selectedSystemId = info.systemId;
            this.view.selectedBodyId = probe.bodyId;
          }
        }
        return;
      }
      case "build-structure":
        this.state = queueStructure(
          this.state,
          command.factionId,
          command.bodyId,
          command.structure,
          this.state.tick
        );
        return;
      case "build-selected": {
        const player = this.state.factions[0];
        if (!player || !this.view.selectedBodyId) {
          return;
        }
        this.state = queueStructure(
          this.state,
          player.id,
          this.view.selectedBodyId,
          command.structure,
          this.state.tick
        );
        return;
      }
      case "set-probe-design": {
        const player = this.state.factions[0];
        if (!player) {
          return;
        }
        this.state = {
          ...this.state,
          factions: this.state.factions.map((faction, index) =>
            index === 0 ? { ...faction, probeDesign: command.design } : faction
          )
        };
        return;
      }
      default:
        return;
    }
  }

  private selectAt(screen: Vector2): void {
    const click = this.toWorld(screen);
    const player = this.state.factions[0];
    const discovered = new Set(player?.discoveredSystems ?? []);
    const systems = listSystems(this.state.galaxy).filter(
      (system) => discovered.size === 0 || discovered.has(system.id)
    );
    let selectedBody: string | null = null;
    let selectedSystem = this.view.selectedSystemId;
    let bestBodyDistance = Number.POSITIVE_INFINITY;
    const time = this.viewTime;

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

    this.view.selectedSystemId = selectedSystem;
    this.view.selectedBodyId = selectedBody;
  }

  private ensureSystemsInView(): void {
    const spacing = getSystemSpacing();
    const margin = 1;
    const rect = this.canvas.getBoundingClientRect();
    const halfWidth = rect.width / 2 / this.view.zoom;
    const halfHeight = rect.height / 2 / this.view.zoom;
    const minX = Math.floor((this.view.camera.x - halfWidth) / spacing) - margin;
    const maxX = Math.floor((this.view.camera.x + halfWidth) / spacing) + margin;
    const minY = Math.floor((this.view.camera.y - halfHeight) / spacing) - margin;
    const maxY = Math.floor((this.view.camera.y + halfHeight) / spacing) + margin;

    let updatedGalaxy = this.state.galaxy;
    for (let x = minX; x <= maxX; x += 1) {
      for (let y = minY; y <= maxY; y += 1) {
        updatedGalaxy = ensureSystem(updatedGalaxy, { x, y });
      }
    }
    if (updatedGalaxy !== this.state.galaxy) {
      this.state = {
        ...this.state,
        galaxy: updatedGalaxy,
        bodyIndex: buildBodyIndex(updatedGalaxy)
      };
    }
  }

  private recordProfile(frameMs: number, simMs: number, renderMs: number, now: number): void {
    if (!this.profiling.enabled) {
      return;
    }
    this.profiling.frames += 1;
    this.profiling.simMs += simMs;
    this.profiling.renderMs += renderMs;
    this.profiling.frameMs += frameMs;
    if (now - this.profiling.lastLog < 1000) {
      return;
    }
    const seconds = (now - this.profiling.lastLog) / 1000;
    const fps = this.profiling.frames / Math.max(0.001, seconds);
    const avgSim = this.profiling.simMs / Math.max(1, this.profiling.frames);
    const avgRender = this.profiling.renderMs / Math.max(1, this.profiling.frames);
    const avgFrame = this.profiling.frameMs / Math.max(1, this.profiling.frames);
    window.console.log(
      `[profile] fps ${fps.toFixed(1)} · sim ${avgSim.toFixed(2)}ms · render ${avgRender.toFixed(2)}ms · frame ${avgFrame.toFixed(2)}ms`
    );
    this.profiling.frames = 0;
    this.profiling.simMs = 0;
    this.profiling.renderMs = 0;
    this.profiling.frameMs = 0;
    this.profiling.lastLog = now;
  }

  private toWorld(screen: Vector2): Vector2 {
    const rect = this.canvas.getBoundingClientRect();
    const localX = screen.x - rect.left;
    const localY = screen.y - rect.top;
    return {
      x: (localX - rect.width / 2) / this.view.zoom + this.view.camera.x,
      y: (localY - rect.height / 2) / this.view.zoom + this.view.camera.y
    };
  }

  private getBodyWorldPosition(bodyId: string): { position: Vector2; systemId: string } | null {
    for (const system of listSystems(this.state.galaxy)) {
      const body = system.bodies.find((item) => item.id === bodyId);
      if (body) {
        const orbit = orbitPosition(body, this.viewTime, { x: 0, y: 0 });
        return {
          position: { x: system.position.x + orbit.x, y: system.position.y + orbit.y },
          systemId: system.id
        };
      }
    }
    return null;
  }
}
