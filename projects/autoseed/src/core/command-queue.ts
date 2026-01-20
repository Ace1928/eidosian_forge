import type { GameCommand } from "./commands.js";

export class CommandQueue {
  private items: GameCommand[] = [];

  enqueue(command: GameCommand): void {
    this.items.push(command);
  }

  drain(): GameCommand[] {
    const drained = this.items;
    this.items = [];
    return drained;
  }

  size(): number {
    return this.items.length;
  }
}
