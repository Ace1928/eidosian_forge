import { GameEngine } from "./game-engine.js";

const config = {
  seed: 4201337,
  systemCount: 7
};

const canvas = document.querySelector<HTMLCanvasElement>("#game");
if (!canvas) {
  throw new Error("Canvas element not found");
}

const engine = new GameEngine(canvas, config);
engine.start();
