from __future__ import annotations

import random
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import tcod

from actions import Action, BumpAction, MeleeAction, MovementAction, WaitAction

if TYPE_CHECKING:
  from entity import Actor

class BaseAI(Action):
  entity: Actor

  def perform(self) -> None:
    raise NotImplementedError

  def get_path_to(self, dest_x: int, dest_y: int) -> List[Tuple[int, int]]:
    cost = np.array(self.entity.gamemap.tiles["walkable"], dtype=np.int8)

    for entity in self.entity.gamemap.entities:
      if entity.blocks_movement and cost[entity.x, entity.y]:
        cost[entity.x, entity.y] = 0

    dist = tcod.path.maxarray((self.entity.gamemap.width, self.entity.gamemap.height), dtype=np.int32)
    dist[self.entity.x, self.entity.y] = 0

    tcod.path.dijkstra2d(dist, cost, 2, 3)

    path = tcod.path.hillclimb2d(dist, (self.engine.player.x, self.engine.player.y), True, True)
    path = path.tolist()
    path.pop()
    return path

class ConfusedEnemy(BaseAI):
  def __init__(
    self, entity: Actor, previous_ai: Optional[BaseAI], turns_remaining: int
  ):
    super().__init__(entity)

    self.previous_ai = previous_ai
    self.turns_remaining = turns_remaining

  def perform(self) -> None:
    if self.turns_remaining <= 0:
      self.engine.message_log.add_message(
        f"The {self.entity.name} is no longer confused."
      )
      self.entity.ai = self.previous_ai
    else:
      direction_x, direction_y = random.choice(
        [
          (-1, -1),  # Northwest
          (0, -1),  # North
          (1, -1),  # Northeast
          (-1, 0),  # West
          (1, 0),  # East
          (-1, 1),  # Southwest
          (0, 1),  # South
          (1, 1),  # Southeast
        ]
      )

      self.turns_remaining -= 1

      return BumpAction(self.entity, direction_x, direction_y).perform()


class HostileEnemy(BaseAI):
  def __init__(self, entity: Actor):
    super().__init__(entity)
    self.path: List[Tuple[int, int]] = []

  def perform(self) -> None:
    target = self.engine.player
    dx = target.x - self.entity.x
    dy = target.y - self.entity.y
    distance = max(abs(dx), abs(dy))

    if self.entity.fighter.activated:
      if distance <= 1:
        return MeleeAction(self.entity, dx, dy).perform()

      self.path = self.get_path_to(target.x, target.y)

    if self.path:
      dest_x, dest_y = self.path.pop()
      return MovementAction(
        self.entity, dest_x - self.entity.x, dest_y - self.entity.y
      ).perform()

    return WaitAction(self.entity).perform()