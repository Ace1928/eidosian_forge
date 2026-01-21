import type {
  Faction,
  GameState,
  ProbeDesign,
  Structure,
  StructureType,
  TechEffect
} from "../core/types.js";
import {
  getExtractorYieldForFaction,
  getStructureBlueprint,
  getStructureCost
} from "../core/simulation.js";
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

const formatTechEffect = (effect: TechEffect): string => {
  const value = effect.value.toFixed(2);
  switch (effect.key) {
    case "mass":
      return `Mass Yield x${value}`;
    case "energy":
      return `Energy Yield x${value}`;
    case "exotic":
      return `Exotic Yield x${value}`;
    case "replication":
      return `Replication x${value}`;
    case "defense":
      return `Defense x${value}`;
    case "attack":
      return `Attack x${value}`;
    case "speed":
      return `Speed x${value}`;
    case "efficiency":
      return `Cost Efficiency x${value}`;
    default:
      return `Effect x${value}`;
  }
};

const tooltipText = {
  tick: "Current simulation tick.",
  status: "Simulation state.",
  speed: "Simulation speed multiplier.",
  zoom: "Camera zoom level.",
  probes: "Active probes under your control.",
  combat: "Systems where multiple factions have active probes.",
  losses: "Probe losses resolved during the most recent tick.",
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
  extractor:
    "Extractor: harvests mass, energy, and exotic per tick. Conflicts with Replicator. Shortcut: 1.",
  replicator: "Replicator: builds new probes over time. Conflicts with Extractor. Shortcut: 2.",
  defense: "Defense: reduces combat losses for your probes in this system. Shortcut: 3."
};

export const updatePanels = (
  state: GameState,
  view: ViewState,
  onBuild: (request: BuildRequest) => void,
  onAction: (action: PanelAction) => void,
  onDesign: (design: ProbeDesign) => void
): void => {
  const player = state.factions[0];
  if (!player) {
    return;
  }

  const systemInfo = byId<HTMLDivElement>("system-info");
  const bodyInfo = byId<HTMLDivElement>("body-info");
  const buildInfo = byId<HTMLDivElement>("build-info");
  const resourceInfo = byId<HTMLDivElement>("resource-info");
  const techInfo = byId<HTMLDivElement>("tech-info");
  const designInfo = byId<HTMLDivElement>("design-info");

  const system = listSystems(state.galaxy).find((sys) => sys.id === view.selectedSystemId) ?? null;
  const selectedBody = system?.bodies.find((body) => body.id === view.selectedBodyId) ?? null;
  const activeProbeCount = player.probes.filter((probe) => probe.active).length;
  const playerProbesInSystem = system
    ? player.probes.filter((probe) => probe.systemId === system.id && probe.active).length
    : 0;
  const probeCountsByFaction = system
    ? state.factions.map(
        (faction) =>
          faction.probes.filter((probe) => probe.active && probe.systemId === system.id).length
      )
    : [];
  const hostileProbesInSystem = probeCountsByFaction.reduce(
    (sum, count, index) => (index === 0 ? sum : sum + count),
    0
  );
  const systemContested = probeCountsByFaction.filter((count) => count > 0).length > 1;
  const playerProbeSystems = new Set(
    player.probes.filter((probe) => probe.active).map((probe) => probe.systemId)
  );
  const contestedCount = state.combat.contestedSystems.filter((systemId) =>
    playerProbeSystems.has(systemId)
  ).length;
  const recentLosses = state.combat.lastTickLosses[player.id] ?? 0;
  const outcomeLabel = state.outcome
    ? state.outcome.winnerId === player.id
      ? "Victory"
      : state.outcome.winnerId
        ? "Defeat"
        : "Stalemate"
    : "";

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
    activeProbeCount,
    player.structures.length,
    player.probeDesign.mining.toFixed(1),
    player.probeDesign.replication.toFixed(1),
    player.probeDesign.defense.toFixed(1),
    player.probeDesign.attack.toFixed(1),
    player.probeDesign.speed.toFixed(1),
    state.outcome?.winnerId ?? "",
    state.outcome?.reason ?? ""
  ].join("|");

  if (key === lastRenderKey) {
    return;
  }
  lastRenderKey = key;

  const statusLabel = outcomeLabel || (view.paused ? "Paused" : "Running");
  resourceInfo.innerHTML = `
    <div class="resource-row">
      <div class="hud-group">
        <span class="hud-tooltip" data-tooltip="${tooltipText.tick}">Tick ${state.tick}</span>
        <span class="hud-tooltip" data-tooltip="${tooltipText.status}">${statusLabel}</span>
      </div>
      <div class="hud-group">
        <span class="hud-tooltip" data-tooltip="${tooltipText.speed}">Speed ${view.speed.toFixed(1)}</span>
        <span class="hud-tooltip" data-tooltip="${tooltipText.zoom}">Zoom ${view.zoom.toFixed(1)}</span>
        <span class="hud-tooltip" data-tooltip="${tooltipText.probes}">Probes ${activeProbeCount}</span>
      </div>
    </div>
    <div class="resource-row">
      <div class="hud-group">
        <span class="hud-tooltip" data-tooltip="${tooltipText.mass}">Mass ${player.resources.mass.toFixed(1)}</span>
        <span class="hud-tooltip" data-tooltip="${tooltipText.energy}">Energy ${player.resources.energy.toFixed(1)}</span>
        <span class="hud-tooltip" data-tooltip="${tooltipText.exotic}">Exotic ${player.resources.exotic.toFixed(1)}</span>
      </div>
      <div class="controls">
        <button data-action="pause" data-tooltip="${tooltipText.pause}" aria-label="Toggle pause">${view.paused ? "Resume" : "Pause"}</button>
        <button data-action="speed-down" data-tooltip="${tooltipText.speedDown}" aria-label="Decrease speed">Slower</button>
        <button data-action="speed-up" data-tooltip="${tooltipText.speedUp}" aria-label="Increase speed">Faster</button>
        <button data-action="zoom-out" data-tooltip="${tooltipText.zoomOut}" aria-label="Zoom out">Zoom -</button>
        <button data-action="zoom-in" data-tooltip="${tooltipText.zoomIn}" aria-label="Zoom in">Zoom +</button>
      </div>
    </div>
    <div class="resource-row">
      <div class="hud-group">
        <span class="hud-tooltip" data-tooltip="${tooltipText.combat}">Combat ${contestedCount}</span>
        <span class="hud-tooltip" data-tooltip="${tooltipText.losses}">Losses ${recentLosses}</span>
      </div>
    </div>
  `;

  const systemDiscovered = system ? player.discoveredSystems.includes(system.id) : false;
  if (system && systemDiscovered) {
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
      <p>Probes in system ${playerProbesInSystem} · Hostiles ${hostileProbesInSystem}</p>
      <p>Combat ${systemContested ? "Contested" : "Clear"}</p>
    `;
  } else if (system) {
    systemInfo.innerHTML = "<h3>Unknown system</h3><p>Deploy a probe to reveal details.</p>";
  } else {
    systemInfo.innerHTML = "<h3>No system selected</h3>";
  }

  const playerTechs = new Set(player.techs);
  const techNames = new Map(state.techTree.nodes.map((node) => [node.id, node.name]));
  const tiers = new Map<number, typeof state.techTree.nodes>();
  state.techTree.nodes.forEach((node) => {
    const existing = tiers.get(node.tier) ?? [];
    existing.push(node);
    tiers.set(node.tier, existing);
  });
  const sortedTiers = [...tiers.entries()].sort(([a], [b]) => a - b);
  techInfo.innerHTML = `
    <h3>Tech Tree</h3>
    ${sortedTiers
      .map(([tier, nodes]) => {
        const entries = nodes
          .map((node) => {
            const unlocked = playerTechs.has(node.id);
            const requirements = node.dependsOn.length
              ? `Requires ${node.dependsOn.map((id) => techNames.get(id) ?? id).join(", ")}`
              : "Starter";
            const effects = node.effects.map((effect) => formatTechEffect(effect)).join(" · ");
            return `
              <div class="tech-node ${unlocked ? "unlocked" : "locked"}">
                <div class="tech-title">${node.name}</div>
                <div class="tech-meta">${unlocked ? "Unlocked" : requirements}</div>
                <div class="tech-effects">${effects}</div>
              </div>
            `;
          })
          .join("");
        return `
          <div class="tech-tier">
            <h4>Tier ${tier}</h4>
            ${entries}
          </div>
        `;
      })
      .join("")}
  `;

  const designTotal =
    player.probeDesign.mining +
    player.probeDesign.replication +
    player.probeDesign.defense +
    player.probeDesign.attack +
    player.probeDesign.speed;
  const designRow = (key: keyof ProbeDesign, label: string, shortcut: string): string => `
    <div class="design-row">
      <label for="design-${key}">${label} <span class="muted">${Math.round(player.probeDesign[key])}%</span></label>
      <input
        id="design-${key}"
        data-design="${key}"
        type="range"
        min="0"
        max="100"
        value="${Math.round(player.probeDesign[key])}"
        step="1"
        aria-label="${label} allocation"
        aria-keyshortcuts="${shortcut}"
      />
    </div>
  `;
  designInfo.innerHTML = `
    <h3>Probe Design</h3>
    <p class="muted">Allocate focus across probe systems. Values are normalized on build.</p>
    ${designRow("mining", "Mining", "Shift+1")}
    ${designRow("replication", "Replication", "Shift+2")}
    ${designRow("defense", "Defense", "Shift+3")}
    ${designRow("attack", "Attack", "Shift+4")}
    ${designRow("speed", "Speed", "Shift+5")}
    <p class="muted">Total ${Math.round(designTotal)}%</p>
  `;

  designInfo.querySelectorAll<HTMLInputElement>("input[data-design]").forEach((input) => {
    input.addEventListener("change", () => {
      const key = input.getAttribute("data-design") as keyof ProbeDesign | null;
      if (!key) {
        return;
      }
      const value = Number(input.value);
      if (!Number.isFinite(value)) {
        return;
      }
      onDesign({ ...player.probeDesign, [key]: value });
    });
  });

  if (selectedBody && systemDiscovered) {
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
  } else if (system && systemDiscovered) {
    bodyInfo.innerHTML = "<h3>Select a body</h3>";
  } else {
    bodyInfo.innerHTML = "";
  }

  const navigation = `
    <div class="build-row">
      <button data-action="center-selected" data-tooltip="${tooltipText.centerSelected}" aria-label="Center on selected body" ${selectedBody && systemDiscovered ? "" : "disabled"}>Center Body</button>
      <button data-action="center-probe" data-tooltip="${tooltipText.centerProbe}" aria-label="Center on probe">Center Probe</button>
    </div>
  `;

  if (!selectedBody || !systemDiscovered) {
    const message = systemDiscovered
      ? "Select a body to construct."
      : "Deploy a probe to unlock construction.";
    buildInfo.innerHTML = `<h3>Build</h3>${navigation}<p>${message}</p>`;
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
    const status = hasType ? "Queued" : conflict ? "Conflict" : "Available";
    const shortcut = type === "extractor" ? "1" : type === "replicator" ? "2" : "3";
    return `
      <button data-structure="${type}" data-tooltip="${structureTooltips[type]}" aria-label="Build ${label}" aria-keyshortcuts="${shortcut}" ${disabled}>${label}</button>
      <span class="muted">${formatCost(adjustedCost)} · ${blueprint.ticks} ticks · ${status}</span>
    `;
  };

  const structureList = existing
    .map(
      (structure) =>
        `${structure.type} ${structure.completed ? "Ready" : formatPercent(structure.progress)}`
    )
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
