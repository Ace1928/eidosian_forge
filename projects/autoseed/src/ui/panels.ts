import type { Faction, GameState, Structure, StructureType } from "../core/types.js";
import { getExtractorYieldForFaction, getStructureBlueprint, getStructureCost } from "../core/simulation.js";
import { listSystems } from "../core/procgen.js";
import type { ViewState } from "./render.js";

export interface BuildRequest {
  factionId: string;
  bodyId: string;
  type: StructureType;
}

const byId = <T extends HTMLElement>(id: string): T => {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`Missing element #${id}`);
  }
  return element as T;
};

const formatCost = (cost: { mass: number; energy: number; exotic: number }): string =>
  `M ${cost.mass}  E ${cost.energy}  X ${cost.exotic}`;

const formatPercent = (value: number): string => `${Math.round(value * 100)}%`;

const structuresOnBody = (faction: Faction, bodyId: string): Structure[] =>
  faction.structures.filter((structure) => structure.bodyId === bodyId);

export type PanelAction =
  | "center-selected"
  | "center-probe"
  | "pause"
  | "speed-up"
  | "speed-down"
  | "zoom-in"
  | "zoom-out";

let lastRenderKey = "";

const tooltipText = {
  tick: "Current simulation tick.",
  status: "Simulation state.",
  speed: "Simulation speed multiplier.",
  zoom: "Camera zoom level.",
  probes: "Active probes under your control.",
  mass: "Mass stockpile. Used for construction and replication.",
  energy: "Energy stockpile. Used for upkeep and construction.",
  exotic: "Exotic stockpile. Required for advanced structures.",
  centerSelected: "Center the camera on the selected body.",
  centerProbe: "Center the camera on your first probe.",
  pause: "Pause or resume the simulation.",
  speedDown: "Decrease the simulation speed.",
  speedUp: "Increase the simulation speed.",
  zoomOut: "Zoom the camera out.",
  zoomIn: "Zoom the camera in."
};

const structureTooltips: Record<StructureType, string> = {
  extractor: "Extractor: harvests mass, energy, and exotic per tick. Conflicts with Replicator.",
  replicator: "Replicator: builds new probes over time. Conflicts with Extractor.",
  defense: "Defense: infrastructure for future combat upgrades."
};

export const updatePanels = (
  state: GameState,
  view: ViewState,
  onBuild: (request: BuildRequest) => void,
  onAction: (action: PanelAction) => void
): void => {
  const player = state.factions[0];
  if (!player) {
    return;
  }

  const systemInfo = byId<HTMLDivElement>("system-info");
  const bodyInfo = byId<HTMLDivElement>("body-info");
  const buildInfo = byId<HTMLDivElement>("build-info");
  const resourceInfo = byId<HTMLDivElement>("resource-info");

  const system = listSystems(state.galaxy).find((sys) => sys.id === view.selectedSystemId) ?? null;
  const selectedBody = system?.bodies.find((body) => body.id === view.selectedBodyId) ?? null;
  const playerProbesInSystem = system
    ? player.probes.filter((probe) => probe.systemId === system.id).length
    : 0;

  const key = [
    state.tick,
    view.paused,
    view.speed.toFixed(2),
    view.zoom.toFixed(2),
    view.selectedSystemId,
    view.selectedBodyId ?? "",
    player.resources.mass.toFixed(1),
    player.resources.energy.toFixed(1),
    player.resources.exotic.toFixed(1),
    player.probes.length,
    player.structures.length
  ].join("|");

  if (key === lastRenderKey) {
    return;
  }
  lastRenderKey = key;

  const statusLabel = view.paused ? "Paused" : "Running";
  resourceInfo.innerHTML = `
    <div class="resource-row">
      <div class="hud-group">
        <span class="hud-tooltip" data-tooltip="${tooltipText.tick}">Tick ${state.tick}</span>
        <span class="hud-tooltip" data-tooltip="${tooltipText.status}">${statusLabel}</span>
      </div>
      <div class="hud-group">
        <span class="hud-tooltip" data-tooltip="${tooltipText.speed}">Speed ${view.speed.toFixed(1)}</span>
        <span class="hud-tooltip" data-tooltip="${tooltipText.zoom}">Zoom ${view.zoom.toFixed(1)}</span>
        <span class="hud-tooltip" data-tooltip="${tooltipText.probes}">Probes ${player.probes.length}</span>
      </div>
    </div>
    <div class="resource-row">
      <div class="hud-group">
        <span class="hud-tooltip" data-tooltip="${tooltipText.mass}">Mass ${player.resources.mass.toFixed(1)}</span>
        <span class="hud-tooltip" data-tooltip="${tooltipText.energy}">Energy ${player.resources.energy.toFixed(1)}</span>
        <span class="hud-tooltip" data-tooltip="${tooltipText.exotic}">Exotic ${player.resources.exotic.toFixed(1)}</span>
      </div>
      <div class="controls">
        <button data-action="pause" data-tooltip="${tooltipText.pause}">${view.paused ? "Resume" : "Pause"}</button>
        <button data-action="speed-down" data-tooltip="${tooltipText.speedDown}">Slower</button>
        <button data-action="speed-up" data-tooltip="${tooltipText.speedUp}">Faster</button>
        <button data-action="zoom-out" data-tooltip="${tooltipText.zoomOut}">Zoom -</button>
        <button data-action="zoom-in" data-tooltip="${tooltipText.zoomIn}">Zoom +</button>
      </div>
    </div>
  `;

  if (system) {
    const byType = system.bodies.reduce(
      (acc, body) => {
        acc[body.type] = (acc[body.type] ?? 0) + 1;
        return acc;
      },
      {} as Record<string, number>
    );
    systemInfo.innerHTML = `
      <h3>${system.name}</h3>
      <p>Star ${system.starClass} · Grid ${system.grid.x}, ${system.grid.y}</p>
      <p>Bodies ${system.bodies.length} · Rocky ${byType.rocky ?? 0} · Gas ${byType.gas ?? 0}</p>
      <p>Ice ${byType.ice ?? 0} · Belts ${byType.belt ?? 0}</p>
      <p>Probes in system ${playerProbesInSystem}</p>
    `;
  } else {
    systemInfo.innerHTML = "<h3>No system selected</h3>";
  }

  if (selectedBody) {
    const yieldRate = getExtractorYieldForFaction(player, state.techTree, selectedBody);
    const probesOnBody = player.probes.filter((probe) => probe.bodyId === selectedBody.id).length;
    bodyInfo.innerHTML = `
      <h3>${selectedBody.name}</h3>
      <p>Type ${selectedBody.type}</p>
      <p>Richness ${selectedBody.properties.richness.toFixed(2)}</p>
      <p>Exotic ${selectedBody.properties.exoticness.toFixed(2)}</p>
      <p>Gravity ${selectedBody.properties.gravity.toFixed(2)}</p>
      <p>Temp ${selectedBody.properties.temperature.toFixed(2)}</p>
      <p>Yield M ${yieldRate.mass.toFixed(2)} · E ${yieldRate.energy.toFixed(2)} · X ${yieldRate.exotic.toFixed(2)}</p>
      <p>Probes on body ${probesOnBody}</p>
    `;
  } else if (system) {
    bodyInfo.innerHTML = "<h3>Select a body</h3>";
  } else {
    bodyInfo.innerHTML = "";
  }

  const navigation = `
    <div class="build-row">
      <button data-action="center-selected" data-tooltip="${tooltipText.centerSelected}" ${selectedBody ? "" : "disabled"}>Center Body</button>
      <button data-action="center-probe" data-tooltip="${tooltipText.centerProbe}">Center Probe</button>
    </div>
  `;

  if (!selectedBody) {
    buildInfo.innerHTML = `<h3>Build</h3>${navigation}<p>Select a body to construct.</p>`;
    [resourceInfo, buildInfo].forEach((panel) => {
      panel.querySelectorAll("button[data-action], .controls button").forEach((button) => {
        button.addEventListener(
          "click",
          () => {
            const action = button.getAttribute("data-action") as PanelAction | null;
            if (action) {
              onAction(action);
            }
          },
          { once: true }
        );
      });
    });
    return;
  }

  const existing = structuresOnBody(player, selectedBody.id);
  const buildButton = (type: StructureType, label: string): string => {
    const blueprint = getStructureBlueprint(type);
    const adjustedCost = getStructureCost(player, state.techTree, type);
    const hasType = existing.some((structure) => structure.type === type);
    const conflict =
      (type === "extractor" && existing.some((structure) => structure.type === "replicator")) ||
      (type === "replicator" && existing.some((structure) => structure.type === "extractor"));
    const disabled = hasType || conflict ? "disabled" : "";
    const status = hasType
      ? "Queued"
      : conflict
        ? "Conflict"
        : "Available";
    return `
      <button data-structure="${type}" data-tooltip="${structureTooltips[type]}" ${disabled}>${label}</button>
      <span class="muted">${formatCost(adjustedCost)} · ${blueprint.ticks} ticks · ${status}</span>
    `;
  };

  const structureList = existing
    .map((structure) => `${structure.type} ${structure.completed ? "Ready" : formatPercent(structure.progress)}`)
    .join(" · ");

  buildInfo.innerHTML = `
    <h3>Build</h3>
    ${navigation}
    <div class="build-row">${buildButton("extractor", "Extractor")}</div>
    <div class="build-row">${buildButton("replicator", "Replicator")}</div>
    <div class="build-row">${buildButton("defense", "Defense")}</div>
    <p class="muted">${structureList || "No structures"}</p>
  `;

  buildInfo.querySelectorAll("button[data-structure]").forEach((button) => {
    button.addEventListener(
      "click",
      () => {
        const type = button.getAttribute("data-structure") as StructureType | null;
        if (!type) {
          return;
        }
        onBuild({ factionId: player.id, bodyId: selectedBody.id, type });
      },
      { once: true }
    );
  });

  [resourceInfo, buildInfo].forEach((panel) => {
    panel.querySelectorAll("button[data-action], .controls button").forEach((button) => {
      button.addEventListener(
        "click",
        () => {
          const action = button.getAttribute("data-action") as PanelAction | null;
          if (action) {
            onAction(action);
          }
        },
        { once: true }
      );
    });
  });
};
