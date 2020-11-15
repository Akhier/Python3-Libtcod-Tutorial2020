from __future__ import annotations

from typing import Callable, Optional, Tuple, TYPE_CHECKING

import tcod

import actions
from actions import (
  Action,
  BumpAction,
  PickupAction,
  WaitAction
)
import color
import exceptions

if TYPE_CHECKING:
  from engine import Engine
  from entity import Item


MOVE_KEYS = {
  # Arrow Keys
  tcod.event.K_UP: (0, -1),
  tcod.event.K_DOWN: (0, 1),
  tcod.event.K_LEFT: (-1, 0),
  tcod.event.K_RIGHT: (1, 0),
  tcod.event.K_HOME: (-1, -1),
  tcod.event.K_END: (-1, 1),
  tcod.event.K_PAGEUP: (1, -1),
  tcod.event.K_PAGEDOWN: (1, 1),
  # Numpad Keys
  tcod.event.K_KP_1: (-1, 1),
  tcod.event.K_KP_2: (0, 1),
  tcod.event.K_KP_3: (1, 1),
  tcod.event.K_KP_4: (-1, 0),
  tcod.event.K_KP_6: (1, 0),
  tcod.event.K_KP_7: (-1, -1),
  tcod.event.K_KP_8: (0, -1),
  tcod.event.K_KP_9: (1, -1),
  # Vi Keys
  tcod.event.K_h: (-1, 0),
  tcod.event.K_j: (0, 1),
  tcod.event.K_k: (0, -1),
  tcod.event.K_l: (1, 0),
  tcod.event.K_y: (-1, -1),
  tcod.event.K_u: (1, -1),
  tcod.event.K_b: (-1, 1),
  tcod.event.K_n: (1, 1)
}

WAIT_KEYS = {
  tcod.event.K_PERIOD,
  tcod.event.K_KP_5,
  tcod.event.K_CLEAR
}

CONFIRM_KEYS = {
  tcod.event.K_RETURN,
  tcod.event.K_KP_ENTER
}


class EventHandler(tcod.event.EventDispatch[Action]):
  def __init__(self, engine:Engine):
    self.engine = engine

  def handle_events(self, event: tcod.event.Event) -> None:
    self.handle_action(self.dispatch(event))

  def handle_action(self, action: Optional[Action]) -> bool:
    if action is None:
      return False

    try:
      action.perform()
    except exceptions.Impossible as exc:
      self.engine.message_log.add_message(exc.args[0], color.impossible)
      return False

    self.engine.handle_enemy_turns()

    self.engine.update_fov()
    return True

  def ev_mousemotion(self, event: tcod.event.MouseMotion) -> None:
    if self.engine.game_map.in_bounds(event.tile.x, event.tile.y):
      self.engine.mouse_location = event.tile.x, event.tile.y

  def ev_quit(self, event: tcod.event.Quit) -> Optional[Action]:
    raise SystemExit()

  def on_render(self, console: tcod.Console) -> None:
    self.engine.render(console)


class AskUserEventHandler(EventHandler):
  """Handles user input for actions which require special input"""

  def handle_action(self, action: Optional[Action]) -> bool:
    if super().handle_action(action):
      self.engine.event_handler = MainGameEventHandler(self.engine)
      return True
    return False

  def ev_keydown(self, event: tcod.event.KeyDown) -> Optional[Action]:
    if event.sym in {
      tcod.event.K_LSHIFT,
      tcod.event.K_RSHIFT,
      tcod.event.K_LCTRL,
      tcod.event.K_RCTRL,
      tcod.event.K_LALT,
      tcod.event.K_RALT
    }:
      return None
    return self.on_exit()

  def ev_mousebuttondown(self, event: tcod.event.MouseButtonDown) -> Optional[Action]:
    return self.on_exit()

  def on_exit(self) -> Optional[Action]:
    self.engine.event_handler = MainGameEventHandler(self.engine)
    return None


class InventoryEventHandler(AskUserEventHandler):
  TITLE = "<missing title>"

  def on_render(self, console: tcod.Console) -> None:
    super().on_render(console)
    number_of_items_in_inventory = len(self.engine.player.inventory.items)

    height = number_of_items_in_inventory + 2

    if height <= 3:
      height = 3

    if self.engine.player.x <= 30:
      x = 40
    else:
      x = 0

    y = 0

    width = len(self.TITLE) + 4

    console.draw_frame(
      x=x,
      y=y,
      width=width,
      height=height,
      title=self.TITLE,
      clear=True,
      fg=(255, 255, 255),
      bg=(0, 0, 0)
    )

    if number_of_items_in_inventory > 0:
      for i, item in enumerate(self.engine.player.inventory.items):
        item_key = chr(ord("a") + i)
        console.print(x + 1, y + i + 1, f"({item_key}) {item.name}")
    else:
      console.print(x + 1, y + 1, "(Empty)")

  def ev_keydown(self, event: tcod.event.KeyDown) -> Optional[Action]:
    player = self.engine.player
    key = event.sym
    index = key - tcod.event.K_a

    if 0 <= index <= 26:
      try:
        selected_item = player.inventory.items[index]
      except IndexError:
        self.engine.message_log.add_message("Invalid entry.", color.invalid)
        return None
      return self.on_item_selected(selected_item)
    return super().ev_keydown(event)

  def on_item_selected(self, item: Item) -> Optional[Action]:
    raise NotImplementedError()


class InventoryActivateHandler(InventoryEventHandler):
  TITLE = "Select an item to use"

  def on_item_selected(self, item: Item) -> Optional[Action]:
    return item.consumable.get_action(self.engine.player)


class InventoryDropHandler(InventoryEventHandler):
  TITLE = "Select an item to drop"

  def on_item_selected(self, item: Item) -> Optional[Action]:
    return actions.DropItem(self.engine.player, item)


class SelectIndexHandler(AskUserEventHandler):
  def __init__(self, engine: Engine):
    super().__init__(engine)
    player = self.engine.player
    engine.mouse_location = player.x, player.y

  def on_render(self, console: tcod.Console) -> None:
    super().on_render(console)
    x, y = self.engine.mouse_location
    console.tiles_rgb["bg"][x, y] = color.white
    console.tiles_rgb["fg"][x, y] = color.black

  def ev_keydown(self, event: tcod.event.KeyDown) -> Optional[Action]:
    key = event.sym
    if key in MOVE_KEYS:
      modifier = 1
      if event.mod & (tcod.event.KMOD_LSHIFT | tcod.event.KMOD_RSHIFT):
        modifier *= 5
      if event.mod & (tcod.event.KMOD_LCTRL | tcod.event.KMOD_RCTRL):
        modifier *= 10
      if event.mod & (tcod.event.KMOD_LALT | tcod.event.KMOD_RALT):
        modifier *= 20

      x, y = self.engine.mouse_location
      dx, dy = MOVE_KEYS[key]
      x += dx * modifier
      y += dy * modifier
      x = max(0, min(x, self.engine.game_map.width - 1))
      y = max(0, min(y, self.engine.game_map.height - 1))
      self.engine.mouse_location = x, y
      return None
    elif key in CONFIRM_KEYS:
      return self.on_index_selected(*self.engine.mouse_location)
    return super().ev_keydown(event)

  def ev_mousebuttondown(self, event: tcod.event.MouseButtonDown) -> Optional[Action]:
    if self.engine.game_map.in_bounds(*event.tile):
      if event.button == 1:
         return self.on_index_selected(*event.tile)
    return super().ev_mousebuttondown(event)

  def on_index_selected(self, x: int, y: int) -> Optional[Action]:
    raise NotImplementedError()


class LookHandler(SelectIndexHandler):
  def on_index_selected(self, x: int, y: int) -> None:
    self.engine.event_handler = MainGameEventHandler(self.engine)


class SingleRangedAttackHandler(SelectIndexHandler):
  def __init__(
    self, engine: Engine, callback, Callable[[Tuple[int, int, int]], Optional[Action]]
  ):
    super().__init__(engine)

    self.callback = callback

  def on_index_selected(self, x: int, y: int) -> Optional[Action]:
    return self.callback((x, y))


class MainGameEventHandler(EventHandler):
  def ev_keydown(self, event: tcod.event.KeyDown) -> Optional[Action]:
    action: Optional[Action] = None

    key = event.sym

    player = self.engine.player

    if key in MOVE_KEYS:
      dx, dy = MOVE_KEYS[key]
      action = BumpAction(player, dx, dy)
    elif key in WAIT_KEYS:
      action = WaitAction(player)

    elif key == tcod.event.K_ESCAPE:
      raise SystemExit()
    elif key == tcod.event.K_v:
      self.engine.event_handler = HistoryViewer(self.engine)

    elif key == tcod.event.K_g:
      action = PickupAction(player)

    elif key == tcod.event.K_i:
      self.engine.event_handler = InventoryActivateHandler(self.engine)
    elif key == tcod.event.K_d:
      self.engine.event_handler = InventoryDropHandler(self.engine)
    elif key == tcod.event.K_SLASH:
      self.engine.event_handler = LookHandler(self.engine)

    # No valid key was pressed
    return action


class GameOverEventHandler(EventHandler):
  def ev_keydown(self, event: tcod.event.KeyDown) -> None:
    if event.sym == tcod.event.K_ESCAPE:
      raise SystemExit()


CURSOR_Y_KEYS = {
  tcod.event.K_UP: -1,
  tcod.event.K_DOWN: 1,
  tcod.event.K_PAGEUP: -10,
  tcod.event.K_PAGEDOWN: 10
}


class HistoryViewer(EventHandler):
  """Print the history on a larger window which can be navigated."""

  def __init__(self, engine: Engine):
    super().__init__(engine)
    self.log_length = len(engine.message_log.messages)
    self.cursor = self.log_length - 1

  def on_render(self, console: tcod.Console) -> None:
    super().on_render(console) # Draw the main state as the background

    log_console = tcod.Console(console.width - 6, console.height - 6)

    # Draw a frame with a custom banner title
    log_console.draw_frame(0, 0, log_console.width, log_console.height)
    log_console.print_box(
      0, 0, log_console.width, 1, "┤Message history├", alignment=tcod.CENTER
    )

    # Render the message log using the cursor parameter
    self.engine.message_log.render_messages(
      log_console,
      1,
      1,
      log_console.width - 2,
      log_console.height - 2,
      self.engine.message_log.messages[: self.cursor + 1]
    )
    log_console.blit(console, 3, 3)

  def ev_keydown(self, event: tcod.event.KeyDown) -> None:
    # Fancy conditional movement to make it feel right
    if event.sym in CURSOR_Y_KEYS:
      adjust = CURSOR_Y_KEYS[event.sym]
      if adjust < 0 and self.cursor == 0:
        # Only move from the top to the bottom when you're on the edge
        self.cursor = self.log_length - 1
      elif adjust > 0 and self.cursor == self.log_length - 1:
        # Same with bottom to top movement
        self.cursor = 0
      else:
        # Otherwise move while staying clamped to the bounds of the history log
        self.cursor = max(0, min(self.cursor + adjust, self.log_length - 1))
    elif event.sym == tcod.event.K_HOME:
      self.cursor = 0
    elif event.sym == tcod.event.K_END:
      self.cursor = self.log_length - 1
    else:
      self.engine.event_handler = MainGameEventHandler(self.engine)