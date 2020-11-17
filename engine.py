from __future__ import annotations

import lzma
import pickle
import numpy as np
from typing import TYPE_CHECKING

from tcod.console import Console
from tcod.map import compute_fov
from tcod import path

import exceptions
from message_log import MessageLog
import render_functions

if TYPE_CHECKING:
  from entity import Actor
  from game_map import GameMap, GameWorld


class Engine:
  game_map: GameMap
  game_world: GameWorld

  def __init__(self, player: Actor):
    self.message_log = MessageLog()
    self.mouse_location = (0, 0)
    self.player = player

  def handle_enemy_turns(self) -> None:
    cost = np.array(self.game_map.tiles["walkable"], dtype=np.int8)

    dist = path.maxarray((self.game_map.width, self.game_map.height), dtype=np.int32)

    dist[self.player.x, self.player.y] = 0
    path.dijkstra2d(dist, cost, 2, 3)
    
    for entity in set(self.game_map.actors) - {self.player}:
      if entity.ai:
        try:
          entity.ai.perform()
        except exceptions.Impossible:
          pass

  def update_fov(self) -> None:
    self.game_map.visible[:] = compute_fov(
      self.game_map.tiles["transparent"],
      (self.player.x, self.player.y),
      radius=8
    )
    self.game_map.explored |= self.game_map.visible

  def render(self, console: Console) -> None:
    self.game_map.render(console)

    self.message_log.render(console=console, x=21, y=45, width=40, height=5)

    render_functions.render_bar(
      console=console,
      current_value=self.player.fighter.hp,
      maximum_value=self.player.fighter.max_hp,
      total_width=20
    )

    render_functions.render_dungeon_level(
      console=console,
      dungeon_level=self.game_world.current_floor,
      location=(0, 47)
    )

    render_functions.render_names_at_mouse_location(
      console=console, x=21, y=44, engine=self
    )

  def save_as(self, filename: str) -> None:
    save_data = lzma.compress(pickle.dumps(self))
    with open(filename, "wb") as f:
      f.write(save_data)