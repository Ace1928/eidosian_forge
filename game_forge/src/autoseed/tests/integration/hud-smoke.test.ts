// @vitest-environment happy-dom
import { describe, expect, it, vi } from "vitest";
import { createInitialState } from "../../src/core/simulation.js";
import { listSystems } from "../../src/core/procgen.js";
import { updatePanels } from "../../src/ui/panels.js";
import type { ProbeDesign } from "../../src/core/types.js";
import { loadHudMarkup } from "../helpers/dom.js";

describe("integration: hud smoke", () => {
  it("renders panel HTML and wires actions using live state", () => {
    loadHudMarkup();

    const state = createInitialState({ seed: 77, systemCount: 3 });
    const system = listSystems(state.galaxy)[0];
    const body = system?.bodies[0];
    if (!system || !body) {
      throw new Error("Expected a system with a body for HUD smoke test.");
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
    const onAction = vi.fn();
    const onDesign = vi.fn();

    updatePanels(state, view, onBuild, onAction, onDesign);

    const resourceInfo = document.getElementById("resource-info");
    const buildInfo = document.getElementById("build-info");
    const designInfo = document.getElementById("design-info");
    expect(resourceInfo?.innerHTML).toContain("Tick");
    expect(buildInfo?.innerHTML).toContain("Build");

    const actionButton = resourceInfo?.querySelector<HTMLButtonElement>("[data-action]");
    actionButton?.dispatchEvent(new Event("click"));
    expect(onAction).toHaveBeenCalled();

    const buildButton = buildInfo?.querySelector<HTMLButtonElement>("[data-structure]");
    buildButton?.dispatchEvent(new Event("click"));
    expect(onBuild).toHaveBeenCalledWith(
      expect.objectContaining({
        bodyId: body.id,
        factionId: state.factions[0]?.id
      })
    );

    const designInput = designInfo?.querySelector<HTMLInputElement>("input[data-design]");
    if (!designInput) {
      throw new Error("Expected design inputs to be rendered.");
    }
    designInput.value = "80";
    designInput.dispatchEvent(new Event("change"));
    expect(onDesign).toHaveBeenCalledWith(
      expect.objectContaining<ProbeDesign>({
        mining: expect.any(Number)
      })
    );
  });
});
