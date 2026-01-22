// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { GameEngine } from "../src/game-engine.js";
import { listSystems } from "../src/core/procgen.js";
import { orbitPosition } from "../src/core/animation.js";
import type { Faction, StarSystem } from "../src/core/types.js";
import type { GameCommand } from "../src/core/commands.js";
import type { CommandQueue } from "../src/core/command-queue.js";
import { installCanvasContextStub } from "./helpers/canvas.js";
import { loadHudMarkup } from "./helpers/dom.js";

describe("game engine", () => {
  let restoreCanvas: (() => void) | null = null;
  let rafCallback: FrameRequestCallback | null = null;

  beforeEach(() => {
    Object.defineProperty(window, "innerWidth", { value: 800, configurable: true });
    Object.defineProperty(window, "innerHeight", { value: 600, configurable: true });
    Object.defineProperty(window, "devicePixelRatio", { value: 1, configurable: true });
    vi.stubGlobal(
      "requestAnimationFrame",
      vi.fn((callback: FrameRequestCallback) => {
        rafCallback = callback;
        return 42;
      })
    );
    vi.stubGlobal("cancelAnimationFrame", vi.fn());
    restoreCanvas = installCanvasContextStub();
  });

  afterEach(() => {
    restoreCanvas?.();
    rafCallback = null;
    document.body.innerHTML = "";
    vi.unstubAllGlobals();
  });

  it("initializes, resizes, and schedules frames", () => {
    const canvas = document.createElement("canvas");
    document.body.append(canvas);

    const engine = new GameEngine(canvas, { seed: 1, systemCount: 2 });
    engine.start();
    engine.start();

    expect(canvas.width).toBe(800);
    expect(canvas.height).toBe(600);
    expect(window.requestAnimationFrame).toHaveBeenCalledTimes(1);
  });

  it("applies basic commands to view state", () => {
    const canvas = document.createElement("canvas");
    document.body.append(canvas);

    const engine = new GameEngine(canvas, { seed: 4, systemCount: 2 });
    const engineAny = engine as unknown as {
      applyCommand: (command: {
        type: string;
        delta?: number | { x: number; y: number };
        value?: number;
      }) => void;
      view: { speed: number; zoom: number; paused: boolean; camera: { x: number; y: number } };
    };

    engineAny.applyCommand({ type: "toggle-pause" });
    expect(engineAny.view.paused).toBe(true);

    engineAny.applyCommand({ type: "speed-change", delta: 10 });
    expect(engineAny.view.speed).toBe(4);

    engineAny.applyCommand({ type: "zoom-change", delta: -10 });
    expect(engineAny.view.zoom).toBe(0.4);

    engineAny.applyCommand({ type: "zoom-set", value: 5 });
    expect(engineAny.view.zoom).toBe(2.6);

    engineAny.applyCommand({ type: "pan-camera", delta: { x: 10, y: -20 } });
    expect(engineAny.view.camera.x).toBeCloseTo(-10 / 2.6);
    expect(engineAny.view.camera.y).toBeCloseTo(20 / 2.6);
  });

  it("runs a frame and updates panels", () => {
    loadHudMarkup();
    const canvas = document.querySelector<HTMLCanvasElement>("#game");
    if (!canvas) {
      throw new Error("Expected a canvas element in HUD markup.");
    }

    const engine = new GameEngine(canvas, { seed: 5, systemCount: 2 });
    const engineAny = engine as unknown as {
      loop: (time: number) => void;
      last: number;
      tickDuration: number;
      view: { speed: number; selectedBodyId: string | null; selectedSystemId: string };
      state: { tick: number; factions: Faction[] };
      commandQueue: CommandQueue;
    };

    engineAny.view.speed = 1;
    const probe = engineAny.state.factions[0]?.probes[0];
    if (probe) {
      engineAny.view.selectedBodyId = probe.bodyId;
      engineAny.view.selectedSystemId = probe.systemId;
    }
    engine.start();
    if (!rafCallback) {
      throw new Error("Expected requestAnimationFrame to be scheduled.");
    }
    rafCallback(engineAny.last + engineAny.tickDuration + 1);
    expect(engineAny.state.tick).toBe(1);
    expect(document.getElementById("resource-info")?.innerHTML).toContain("Tick");

    const buildButton = document.querySelector<HTMLButtonElement>(
      '#build-info button[data-structure="extractor"]'
    );
    buildButton?.dispatchEvent(new Event("click"));
    const centerProbeButton = document.querySelector<HTMLButtonElement>(
      '#build-info button[data-action="center-probe"]'
    );
    centerProbeButton?.dispatchEvent(new Event("click"));
    const designInput = document.querySelector<HTMLInputElement>('input[data-design="mining"]');
    if (designInput) {
      designInput.value = "70";
      designInput.dispatchEvent(new Event("change"));
    }
    expect(engineAny.commandQueue.size()).toBeGreaterThan(0);

    engineAny.state.outcome = {
      winnerId: engineAny.state.factions[0]?.id ?? null,
      reason: "elimination"
    };
    rafCallback(engineAny.last + engineAny.tickDuration + 1);
    expect(engineAny.view.paused).toBe(true);
  });

  it("selects bodies and centers the camera via commands", () => {
    loadHudMarkup();
    const canvas = document.querySelector<HTMLCanvasElement>("#game");
    if (!canvas) {
      throw new Error("Expected a canvas element in HUD markup.");
    }
    canvas.getBoundingClientRect = () =>
      ({
        left: 0,
        top: 0,
        width: 800,
        height: 600
      }) as DOMRect;

    const engine = new GameEngine(canvas, { seed: 6, systemCount: 2 });
    const engineAny = engine as unknown as {
      applyCommand: (command: { type: string; screen?: { x: number; y: number } }) => void;
      getBodyWorldPosition: (bodyId: string) => { position: { x: number; y: number } } | null;
      view: { camera: { x: number; y: number }; zoom: number; selectedBodyId: string | null };
      viewTime: number;
      state: {
        galaxy: { systems: Record<string, StarSystem> };
        factions: Faction[];
      };
    };

    engineAny.viewTime = 0;
    const system = listSystems(engineAny.state.galaxy)[0];
    const body = system?.bodies[0];
    if (!system || !body) {
      throw new Error("Expected a body for selection test.");
    }
    const orbit = orbitPosition(body, engineAny.viewTime, { x: 0, y: 0 });
    const world = { x: system.position.x + orbit.x, y: system.position.y + orbit.y };
    const screen = {
      x: 400 + (world.x - engineAny.view.camera.x),
      y: 300 + (world.y - engineAny.view.camera.y)
    };

    engineAny.applyCommand({ type: "select-at", screen });
    expect(engineAny.view.selectedBodyId).toBe(body.id);
    expect(engineAny.getBodyWorldPosition("missing-body")).toBeNull();

    const expectedCenter = engineAny.getBodyWorldPosition(body.id);
    if (!expectedCenter) {
      throw new Error("Expected a body world position for centering.");
    }
    engineAny.applyCommand({ type: "center-selected" });
    expect(engineAny.view.camera).toEqual(expectedCenter.position);

    const probeBodyId = engineAny.state.factions[0]?.probes[0]?.bodyId;
    engineAny.applyCommand({ type: "center-probe" });
    expect(engineAny.view.selectedBodyId).toBe(probeBodyId);

    engineAny.view.selectedBodyId = null;
    engineAny.applyCommand({ type: "center-selected" });

    engineAny.view.selectedBodyId = "missing-body";
    engineAny.applyCommand({ type: "center-selected" });

    engineAny.state = {
      ...engineAny.state,
      factions: engineAny.state.factions.map((faction, index) =>
        index === 0 ? { ...faction, probes: [] } : faction
      )
    };
    engineAny.applyCommand({ type: "center-probe" });

    engineAny.state = {
      ...engineAny.state,
      factions: engineAny.state.factions.map((faction, index) =>
        index === 0
          ? {
              ...faction,
              probes: [
                {
                  id: "probe-missing",
                  ownerId: faction.id,
                  systemId: system.id,
                  bodyId: "missing-body",
                  stats: { mining: 1, replication: 1, defense: 1, attack: 1, speed: 1 },
                  active: true
                }
              ]
            }
          : faction
      )
    };
    engineAny.applyCommand({ type: "center-probe" });

    const systemScreen = {
      x: 400 + (system.position.x - engineAny.view.camera.x),
      y: 300 + (system.position.y - engineAny.view.camera.y)
    };
    engineAny.applyCommand({ type: "select-at", screen: systemScreen });
    expect(engineAny.view.selectedBodyId).toBeNull();
  });

  it("queues structures and updates probe design", () => {
    const canvas = document.createElement("canvas");
    document.body.append(canvas);
    const engine = new GameEngine(canvas, { seed: 8, systemCount: 2 });
    const engineAny = engine as unknown as {
      applyCommand: (command: GameCommand) => void;
      state: { factions: Faction[]; tick: number };
      view: { selectedBodyId: string | null };
    };

    const bodyId = engineAny.state.factions[0]?.probes[0]?.bodyId;
    if (!bodyId) {
      throw new Error("Expected a probe body for build test.");
    }
    engineAny.view.selectedBodyId = bodyId;
    engineAny.applyCommand({ type: "build-selected", structure: "extractor" });
    expect(engineAny.state.factions[0]?.structures.length).toBeGreaterThan(0);

    const beforeCount = engineAny.state.factions[0]?.structures.length ?? 0;
    engineAny.view.selectedBodyId = null;
    engineAny.applyCommand({ type: "build-selected", structure: "defense" });
    expect(engineAny.state.factions[0]?.structures.length).toBe(beforeCount);

    engineAny.applyCommand({
      type: "build-structure",
      factionId: engineAny.state.factions[0]?.id,
      bodyId,
      structure: "defense"
    });
    expect(engineAny.state.factions[0]?.structures.length).toBeGreaterThan(1);

    engineAny.applyCommand({
      type: "set-probe-design",
      design: { mining: 20, replication: 20, defense: 20, attack: 20, speed: 20 }
    });
    expect(engineAny.state.factions[0]?.probeDesign.mining).toBe(20);

    const emptyState = { ...engineAny.state, factions: [] };
    engineAny.state = emptyState;
    engineAny.applyCommand({
      type: "set-probe-design",
      design: { mining: 10, replication: 10, defense: 10, attack: 10, speed: 10 }
    });
    engineAny.applyCommand({ type: "unknown-command" } as GameCommand);
  });

  it("expands systems in view and records profiling stats", () => {
    const canvas = document.createElement("canvas");
    document.body.append(canvas);
    canvas.getBoundingClientRect = () =>
      ({
        left: 0,
        top: 0,
        width: 800,
        height: 600
      }) as DOMRect;

    const engine = new GameEngine(canvas, { seed: 10, systemCount: 2 });
    const engineAny = engine as unknown as {
      ensureSystemsInView: () => void;
      recordProfile: (frameMs: number, simMs: number, renderMs: number, now: number) => void;
      profiling: { enabled: boolean; lastLog: number };
      view: { camera: { x: number; y: number } };
      state: { galaxy: { systems: Record<string, unknown> } };
    };

    const before = Object.keys(engineAny.state.galaxy.systems).length;
    engineAny.view.camera = { x: 5000, y: 5000 };
    engineAny.ensureSystemsInView();
    const after = Object.keys(engineAny.state.galaxy.systems).length;
    expect(after).toBeGreaterThan(before);

    const logSpy = vi.spyOn(window.console, "log").mockImplementation(() => undefined);
    engineAny.profiling.enabled = true;
    engineAny.profiling.lastLog = 0;
    engineAny.recordProfile(2, 1, 1, 500);
    engineAny.recordProfile(10, 4, 6, 2000);
    expect(logSpy).toHaveBeenCalled();
    logSpy.mockRestore();
  });

  it("processes queued commands and camera input", () => {
    const canvas = document.createElement("canvas");
    document.body.append(canvas);

    const engine = new GameEngine(canvas, { seed: 12, systemCount: 2 });
    const engineAny = engine as unknown as {
      commandQueue: CommandQueue;
      enqueue: (command: GameCommand) => void;
      processCommands: () => void;
      applyCameraKeys: (delta: number) => void;
      view: { camera: { x: number; y: number }; zoom: number };
      input: { keys: Set<string> };
    };

    engineAny.enqueue({ type: "toggle-pause" });
    expect(engineAny.commandQueue.size()).toBe(1);
    engineAny.commandQueue.enqueue({ type: "zoom-set", value: 2 });
    engineAny.commandQueue.enqueue({ type: "toggle-pause" });
    engineAny.processCommands();
    expect(engineAny.view.zoom).toBe(2);

    const startX = engineAny.view.camera.x;
    const startY = engineAny.view.camera.y;
    engineAny.input.keys.add("d");
    engineAny.input.keys.add("arrowright");
    engineAny.applyCameraKeys(100);
    expect(engineAny.view.camera.x).toBeGreaterThan(startX);

    const afterRight = engineAny.view.camera.x;
    engineAny.input.keys.clear();
    engineAny.input.keys.add("s");
    engineAny.input.keys.add("arrowdown");
    engineAny.applyCameraKeys(100);
    expect(engineAny.view.camera.y).toBeGreaterThan(startY);

    engineAny.input.keys.clear();
    engineAny.input.keys.add("a");
    engineAny.input.keys.add("arrowleft");
    engineAny.applyCameraKeys(100);
    expect(engineAny.view.camera.x).toBeLessThan(afterRight);

    const afterLeft = engineAny.view.camera.y;
    engineAny.input.keys.clear();
    engineAny.input.keys.add("w");
    engineAny.input.keys.add("arrowup");
    engineAny.applyCameraKeys(100);
    expect(engineAny.view.camera.y).toBeLessThan(afterLeft);
  });

  it("returns early when the loop runs without a scheduled frame", () => {
    const canvas = document.createElement("canvas");
    document.body.append(canvas);

    const engine = new GameEngine(canvas, { seed: 13, systemCount: 1 });
    const engineAny = engine as unknown as {
      loop: (time: number) => void;
      last: number;
      frameRequest: number | null;
    };

    expect(engineAny.frameRequest).toBeNull();
    engineAny.loop(100);
    expect(engineAny.last).toBe(0);
  });

  it("falls back to a default pixel ratio when none is available", () => {
    const canvas = document.createElement("canvas");
    document.body.append(canvas);

    const engine = new GameEngine(canvas, { seed: 14, systemCount: 1 });
    const engineAny = engine as unknown as { resize: () => void };

    Object.defineProperty(window, "devicePixelRatio", { value: undefined, configurable: true });
    engineAny.resize();
    expect(canvas.width).toBe(800);
    Object.defineProperty(window, "devicePixelRatio", { value: 1, configurable: true });
  });

  it("handles selection when no player faction exists", () => {
    const canvas = document.createElement("canvas");
    document.body.append(canvas);
    canvas.getBoundingClientRect = () =>
      ({
        left: 0,
        top: 0,
        width: 800,
        height: 600
      }) as DOMRect;

    const engine = new GameEngine(canvas, { seed: 15, systemCount: 1 });
    const engineAny = engine as unknown as {
      selectAt: (screen: { x: number; y: number }) => void;
      state: { factions: unknown[] };
      view: { selectedBodyId: string | null };
    };

    engineAny.state = { ...engineAny.state, factions: [] };
    engineAny.selectAt({ x: 10, y: 10 });
    expect(engineAny.view.selectedBodyId).toBeNull();
  });

  it("cancels frames on destroy", () => {
    const canvas = document.createElement("canvas");
    document.body.append(canvas);

    const engine = new GameEngine(canvas, { seed: 2, systemCount: 2 });
    engine.start();
    engine.destroy();

    expect(window.cancelAnimationFrame).toHaveBeenCalledWith(42);
  });

  it("allows destroy before the loop starts", () => {
    const canvas = document.createElement("canvas");
    document.body.append(canvas);

    const engine = new GameEngine(canvas, { seed: 11, systemCount: 2 });
    engine.destroy();

    expect(window.cancelAnimationFrame).not.toHaveBeenCalled();
  });

  it("throws when canvas context is missing", () => {
    const canvas = document.createElement("canvas");
    document.body.append(canvas);
    const original = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = () => null;

    expect(() => new GameEngine(canvas, { seed: 3, systemCount: 2 })).toThrow(
      "Canvas context not available"
    );

    HTMLCanvasElement.prototype.getContext = original;
  });
});
