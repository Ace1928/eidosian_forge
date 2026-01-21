// @vitest-environment happy-dom
import { describe, expect, it, vi } from "vitest";
import { createInitialState } from "../src/core/simulation.js";
import { listSystems } from "../src/core/procgen.js";
import { updatePanels } from "../src/ui/panels.js";
import { loadHudMarkup } from "./helpers/dom.js";
import type { TechEffect } from "../src/core/types.js";

describe("ui panels", () => {
  const noop = () => undefined;

  it("renders empty selection state", () => {
    loadHudMarkup();
    const state = createInitialState({ seed: 4, systemCount: 2 });
    const onAction = vi.fn();
    const view = {
      selectedSystemId: "",
      selectedBodyId: null,
      paused: false,
      speed: 1,
      camera: { x: 0, y: 0 },
      zoom: 1
    };

    updatePanels(state, view, noop, onAction, noop);
    updatePanels(state, view, noop, onAction, noop);
    expect(document.getElementById("system-info")?.textContent).toContain("No system selected");

    const centerProbe = document.querySelector<HTMLButtonElement>(
      "#build-info button[data-action=\"center-probe\"]"
    );
    centerProbe?.dispatchEvent(new Event("click"));
    expect(onAction).toHaveBeenCalledWith("center-probe");

    const pauseButton = document.querySelector<HTMLButtonElement>(
      "#resource-info button[data-action=\"pause\"]"
    );
    pauseButton?.removeAttribute("data-action");
    pauseButton?.dispatchEvent(new Event("click"));
    expect(onAction).toHaveBeenCalledTimes(1);
  });

  it("throws when required HUD elements are missing", () => {
    loadHudMarkup();
    document.getElementById("resource-info")?.remove();
    const state = createInitialState({ seed: 5, systemCount: 2 });
    const view = {
      selectedSystemId: "",
      selectedBodyId: null,
      paused: false,
      speed: 1,
      camera: { x: 0, y: 0 },
      zoom: 1
    };

    expect(() => updatePanels(state, view, noop, noop, noop)).toThrow(
      "Missing element #resource-info"
    );
  });

  it("renders energy and fallback tech effects", () => {
    loadHudMarkup();
    const state = createInitialState({ seed: 13, systemCount: 2 });
    state.tick = 999;
    const view = {
      selectedSystemId: "",
      selectedBodyId: null,
      paused: false,
      speed: 1,
      camera: { x: 0, y: 0 },
      zoom: 1
    };
    const fallbackEffect = { key: "mystery", value: 0.9 } as unknown as TechEffect;
    state.techTree = {
      ...state.techTree,
      nodes: [
        {
          id: "tech-energy",
          name: "Energy Focus",
          tier: 1,
          description: "Boost energy",
          effects: [{ key: "energy", value: 1.3 }, fallbackEffect],
          dependsOn: []
        }
      ]
    };

    updatePanels(state, view, noop, noop, noop);
    const techInfo = document.getElementById("tech-info")?.innerHTML ?? "";
    expect(techInfo).toContain("Energy Yield x1.30");
    expect(techInfo).toContain("Effect x0.90");
  });

  it("renders system counts with missing body types and tech requirements", () => {
    loadHudMarkup();
    const state = createInitialState({ seed: 14, systemCount: 1 });
    const system = listSystems(state.galaxy)[0];
    if (!system) {
      throw new Error("Missing system");
    }
    system.bodies = [
      {
        id: `${system.id}-rocky-0`,
        name: "Only Rocky",
        type: "rocky",
        systemId: system.id,
        orbitIndex: 0,
        properties: {
          richness: 0.5,
          exoticness: 0.2,
          gravity: 0.4,
          temperature: 0.6,
          size: 0.8
        }
      }
    ];
    state.tick = 1000;
    state.techTree = {
      ...state.techTree,
      nodes: [
        {
          id: "tech-missing",
          name: "Missing Tech",
          tier: 2,
          description: "Needs unknown prereq",
          effects: [{ key: "mass", value: 1.1 }],
          dependsOn: ["tech-ghost"]
        }
      ]
    };

    const view = {
      selectedSystemId: system.id,
      selectedBodyId: null,
      paused: false,
      speed: 1,
      camera: { x: 0, y: 0 },
      zoom: 1
    };

    updatePanels(state, view, noop, noop, noop);
    const systemInfo = document.getElementById("system-info")?.innerHTML ?? "";
    expect(systemInfo).toContain("Rocky 1");
    expect(systemInfo).toContain("Gas 0");
    expect(systemInfo).toContain("Ice 0");
    expect(systemInfo).toContain("Belts 0");

    const techInfo = document.getElementById("tech-info")?.innerHTML ?? "";
    expect(techInfo).toContain("Requires tech-ghost");
  });

  it("renders undiscovered system state", () => {
    loadHudMarkup();
    const state = createInitialState({ seed: 7, systemCount: 3 });
    const systems = listSystems(state.galaxy);
    const undiscovered = systems.find(
      (system) => !state.factions[0]?.discoveredSystems.includes(system.id)
    );
    if (!undiscovered) {
      throw new Error("Expected undiscovered system.");
    }

    const view = {
      selectedSystemId: undiscovered.id,
      selectedBodyId: null,
      paused: false,
      speed: 1,
      camera: { x: 0, y: 0 },
      zoom: 1
    };

    updatePanels(state, view, noop, noop, noop);
    expect(document.getElementById("system-info")?.textContent).toContain("Unknown system");
    expect(document.getElementById("build-info")?.textContent).toContain("Deploy a probe");
  });

  it("prompts to select a body when system is discovered", () => {
    loadHudMarkup();
    const state = createInitialState({ seed: 8, systemCount: 2 });
    const system = listSystems(state.galaxy)[0];
    if (!system) {
      throw new Error("Missing system");
    }

    const view = {
      selectedSystemId: system.id,
      selectedBodyId: null,
      paused: false,
      speed: 1,
      camera: { x: 0, y: 0 },
      zoom: 1
    };

    updatePanels(state, view, noop, noop, noop);
    expect(document.getElementById("build-info")?.textContent).toContain("Select a body");
  });

  it("returns early when no player faction is present", () => {
    loadHudMarkup();
    const state = { ...createInitialState({ seed: 6, systemCount: 2 }), factions: [] };
    const view = {
      selectedSystemId: "",
      selectedBodyId: null,
      paused: false,
      speed: 1,
      camera: { x: 0, y: 0 },
      zoom: 1
    };

    updatePanels(state, view, noop, noop, noop);
    expect(document.getElementById("resource-info")?.textContent).toContain("Mass");
  });

  it("renders conflict state for existing structures", () => {
    loadHudMarkup();
    const state = createInitialState({ seed: 9, systemCount: 2 });
    const system = listSystems(state.galaxy)[0];
    const body = system?.bodies[0];
    const player = state.factions[0];
    if (!system || !body || !player) {
      throw new Error("Missing initial system or body.");
    }

    const nextState = {
      ...state,
      factions: state.factions.map((faction, index) =>
        index === 0
          ? {
              ...faction,
              structures: [
                ...faction.structures,
                {
                  id: "replicator-test",
                  type: "replicator",
                  bodyId: body.id,
                  ownerId: faction.id,
                  progress: 0,
                  completed: false
                },
                {
                  id: "extractor-ready",
                  type: "extractor",
                  bodyId: body.id,
                  ownerId: faction.id,
                  progress: 1,
                  completed: true
                }
              ]
            }
          : faction
      )
    };

    const view = {
      selectedSystemId: system.id,
      selectedBodyId: body.id,
      paused: false,
      speed: 1,
      camera: { x: 0, y: 0 },
      zoom: 1
    };

    updatePanels(nextState, view, noop, noop, noop);
    expect(document.getElementById("build-info")?.innerHTML).toContain("Conflict");
    expect(document.getElementById("build-info")?.innerHTML).toContain("replicator 0%");
    expect(document.getElementById("build-info")?.innerHTML).toContain("extractor Ready");
  });

  it("handles build button clicks and missing data attributes", () => {
    loadHudMarkup();
    const state = createInitialState({ seed: 10, systemCount: 2 });
    const system = listSystems(state.galaxy)[0];
    const body = system?.bodies[0];
    if (!system || !body) {
      throw new Error("Missing system or body");
    }
    const view = {
      selectedSystemId: system.id,
      selectedBodyId: body.id,
      paused: false,
      speed: 1,
      camera: { x: 0, y: 0 },
      zoom: 1
    };
    const onBuild = vi.fn();
    updatePanels(state, view, onBuild, noop, noop);

    const buildButton = document.querySelector<HTMLButtonElement>(
      "#build-info button[data-structure=\"extractor\"]"
    );
    buildButton?.dispatchEvent(new Event("click"));
    expect(onBuild).toHaveBeenCalled();

    const invalidButton = document.querySelector<HTMLButtonElement>(
      "#build-info button[data-structure=\"defense\"]"
    );
    invalidButton?.removeAttribute("data-structure");
    invalidButton?.dispatchEvent(new Event("click"));
    expect(onBuild).toHaveBeenCalledTimes(1);
  });

  it("handles design inputs safely", () => {
    loadHudMarkup();
    const state = createInitialState({ seed: 11, systemCount: 2 });
    const system = listSystems(state.galaxy)[0];
    const body = system?.bodies[0];
    if (!system || !body) {
      throw new Error("Missing system or body");
    }
    const view = {
      selectedSystemId: system.id,
      selectedBodyId: body.id,
      paused: false,
      speed: 1,
      camera: { x: 0, y: 0 },
      zoom: 1
    };
    const onDesign = vi.fn();
    updatePanels(state, view, noop, noop, onDesign);

    const input = document.querySelector<HTMLInputElement>("input[data-design]");
    if (!input) {
      throw new Error("Missing design input");
    }
    Object.defineProperty(input, "value", { value: "NaN", configurable: true });
    input.dispatchEvent(new Event("change"));
    expect(onDesign).not.toHaveBeenCalled();
    Reflect.deleteProperty(input, "value");

    input.removeAttribute("data-design");
    input.value = "50";
    input.dispatchEvent(new Event("change"));
    expect(onDesign).not.toHaveBeenCalled();

    input.setAttribute("data-design", "mining");
    input.value = "70";
    input.dispatchEvent(new Event("change"));
    expect(onDesign).toHaveBeenCalled();
  });

  it("renders outcome states", () => {
    loadHudMarkup();
    const state = createInitialState({ seed: 12, systemCount: 2 });
    const system = listSystems(state.galaxy)[0];
    if (!system) {
      throw new Error("Missing system");
    }
    const view = {
      selectedSystemId: system.id,
      selectedBodyId: null,
      paused: false,
      speed: 1,
      camera: { x: 0, y: 0 },
      zoom: 1
    };

    updatePanels({ ...state, outcome: { winnerId: state.factions[0]?.id ?? null, reason: "elimination" } }, view, noop, noop, noop);
    expect(document.getElementById("resource-info")?.textContent).toContain("Victory");

    updatePanels({ ...state, outcome: { winnerId: "other", reason: "elimination" } }, view, noop, noop, noop);
    expect(document.getElementById("resource-info")?.textContent).toContain("Defeat");

    updatePanels({ ...state, outcome: { winnerId: null, reason: "stalemate" } }, view, noop, noop, noop);
    expect(document.getElementById("resource-info")?.textContent).toContain("Stalemate");
  });
});
