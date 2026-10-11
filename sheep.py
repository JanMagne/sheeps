# Jan Magne Kampen.
"""Sheep herding simulation, built on the boids flocking rules.

Sheep and their lambs graze, drift around the pasture in small groups and
bunch up when the dog comes near. Each sheep has a hunger: it eats grass when
hungry enough (the grass wears down and slowly regrows) but prefers hay.
Starving sheep bleat, slow down and lose wool. Days pass; at night the herd
bunches up and lies down to sleep. The goal is to get the herd into the pen
in the bottom-right corner.

Controls: left click puts out a hay bale (click it again to take it away);
right click lets the sheepdog into the field or moves it (right click the dog
to call it off). G opens and closes the pen gate, H shows bars of how full
each sheep is, R starts again, Esc quits.
"""

import math
import random

import pygame
from pygame import Vector2

# Display size. Change these to resize the window.
SCREEN_RES_X = 2200
SCREEN_RES_Y = 1200
FPS = 60
DT = 1 / FPS  # Fixed simulation step in seconds; keeps behaviour frame-rate independent of lag.

# Herd size and movement. Speeds are in pixels/second, forces in pixels/second².
NUM_SHEEP = 140
NUM_LAMBS = 24  # Each follows one of the sheep, its mother.
GRAZE_SPEED = 3  # Slow amble while eating.
WALK_SPEED = 32  # Moving on to fresh grass.
RUN_SPEED = 170  # Fully frightened.
MAX_SPEED = 45  # Speed cap when calm.
PANIC_SPEED = 210  # Speed cap when fully alarmed.
SPEED_GAIN = 2.0  # How quickly sheep settle to their goal speed (per second).
MAX_FORCE = 80  # Steering limit when calm: radius ≈ speed² / force.
PANIC_FORCE = 320  # Steering limit when fully alarmed, so running sheep can still turn.
TURN_RATE = 300  # Degrees/second the body can turn towards where it is moving.
FACE_SPEED = 6  # Below this speed a sheep steps without turning its body...
FACE_FULL_SPEED = 22  # ...and from this speed up it turns at the full TURN_RATE.

# Grazing rhythm: each sheep switches between grazing and walking at random.
GRAZE_TIME = 14  # Mean seconds spent grazing in one spot.
WALK_TIME = 3  # Mean seconds of walking before grazing again.
FOLLOW = 4  # Sheep are this much more likely to start walking if all neighbours are.

# Perception.
VIEW_RADIUS = 80  # Neighbours further away than this are ignored.
VIEW_ANGLE = 300  # Field of view in degrees; sheep see almost all the way round.
SEPARATION_RADIUS = 26  # Neighbours closer than this are pushed away.

# Rule weights.
SEPARATION_FORCE = 220  # Push from a neighbour right on top of us; fades to 0 at SEPARATION_RADIUS.
ALIGNMENT_GAIN = 0.8  # Pull towards neighbours' mean velocity (per second).
COHESION_GAIN = 0.4  # Pull towards neighbours' centre (per second² per pixel).
ALARM_COHESION = 6  # Cohesion is this much stronger when alarmed: the "selfish herd".
WALK_COHESION = 0.5  # Walkers feel less pull back to the group, so they don't U-turn into it.
WANDER_FORCE = 8  # Sideways drift while moving; scaled down with goal speed.
GRAZE_TURN = 15  # Degrees/second a grazing sheep slowly turns as it feeds.
WANDER_TIME = 3.0  # Seconds the drift keeps roughly the same direction.

# Fear spreads through the herd: a sheep is at least ALARM_SPREAD times as
# alarmed as its most alarmed neighbour, and calms down over ALARM_TIME.
ALARM_SPREAD = 0.8
ALARM_TIME = 3.0

# The pasture is fenced in FENCE_INSET from the window edge. Sheep start
# turning back when they come within EDGE_MARGIN of the fence.
FENCE_INSET = 30
EDGE_MARGIN = 80
EDGE_FORCE = 200

# The pen in the bottom-right corner, with a gate in its left side.
PEN_W = 380
PEN_H = 300
GATE_W = 110
WALL_MARGIN = 8  # Sheep are pushed away from pen walls within this distance of touching...
WALL_PUSH = 200  # ...growing to this force at contact.

# Hay.
FOOD_WEIGHT = 1.0
FOOD_RADIUS = 380  # Sheep only notice hay closer than this...
HUNGER_SIGHT = 1.0  # ...stretched by this fraction of their hunger: starving sheep look twice as far.
FOOD_FADE = 60  # The pull fades out over the last FOOD_FADE pixels of the radius.
EAT_GAP = 5  # Sheep eat within this distance beyond avoid_hay()'s push zone (see Bale.reach).
HAY_RADIUS = 19  # Size of a full bale; clicking within a bale removes it.
HAY_MIN_RADIUS = 9  # Size of a bale that's almost eaten.
BALE_HAY = 900  # Sheep-seconds of eating in a full bale (about 45 meals).
RING_SPOTS = 12  # Places to eat round a bale...
SPOT_CLEAR = 14  # ...each free while no sheep is closer to it than this.
SHEEP_RADIUS = 13  # Body size for bumping into bales and walls; the head still reaches over to eat.
BLOCKED_SPEED = 10  # A neighbour slower than this, ahead of us towards the hay...
QUEUE_COS = 0.7  # ...within ~45° of the way to it, blocks us.
HAY_TURN = 120  # Degrees/second a sheep stopped at a bale turns to face it.
HAY_SWAY = 20  # Degrees either side of the bale it may face, drifting with its wander.
HAY_MARGIN = 5  # Sheep are pushed away from a bale within this distance of touching it...
HAY_PUSH = 300  # ...growing to this force at contact.

# Hunger, 0..1. It rises slowly all the time and falls while eating. A sheep
# that eats hay down to 0 is full: it walks away from the hay and ignores it
# until its hunger is back up to HUNGRY_AT.
HUNGER_RATE = 1 / 180  # Per second: from full to starving in 3 minutes.
EAT_RATE = 1 / 20  # Per second while eating hay: starving to full in ~20 s.
HUNGRY_AT = 0.25
LEAVE_DISTANCE = 70  # Full sheep closer than this to a bale walk away from it.

# Grass: the field is split into GRASS_CELL squares, each with 0..1 grass.
# Sheep with no hay in sight start eating grass at GRASS_HUNGER and stop at
# GRASS_SATED. It fills them more slowly than hay, so they prefer hay.
GRASS_CELL = 20
GRASS_HUNGER = 0.5
GRASS_SATED = 0.2
GRASS_EAT_RATE = 1 / 60  # Hunger lost per second of grazing, a third of hay's.
GRASS_BITE = 1 / 30  # Grass a sheep eats from its cell per second.
GRASS_MIN = 0.15  # Cells with less than this are bare: the sheep walks to greener grass.
GRASS_REGROW = 1 / 300  # Per second: bare to full in 5 minutes.
GRASS_LOOK = (60, 120, 200)  # Distances a hungry sheep looks for greener grass at.
GRASS_TICK = 1.0  # Seconds between regrowth updates.

# Starving: above STARVING hunger sheep bleat, slow down and lose wool, which
# grows back once they're fed (hunger below WOOL_REGROW_BELOW).
STARVING = 0.9
STARVE_SLOW = 0.4  # Fraction of speed lost at full hunger.
BLEAT_RATE = 0.12  # Bleats per second while starving.
BLEAT_TIME = 1.5  # Seconds a speech bubble stays up.
WOOL_LOSS = 1 / 90  # Per second while starving.
WOOL_REGROW = 1 / 150  # Per second while well fed.
WOOL_REGROW_BELOW = 0.5
WOOL_MIN = 0.3

# Sheep don't like being alone: one with no other sheep within VIEW_RADIUS
# walks to the nearest one it can find within LONELY_RADIUS (further at dusk).
LONELY_RADIUS = 320

# Lambs: smaller, and they trot after their mother when she gets too far.
LAMB_SIZE = 0.6
LAMB_STAY = 45  # A lamb further than this from its mother goes after her...
LAMB_CLOSE = 28  # ...until it's this close again.
LAMB_TROT = 42
LAMB_LOST = 130  # Further than this and it bleats for her.
LAMB_BLEAT_RATE = 0.4

# Day and night. A whole day takes DAY_LENGTH seconds. It's dark from
# NIGHT_START to NIGHT_END, with DUSK_HOURS of twilight either side.
DAY_LENGTH = 360
START_HOUR = 8.0
NIGHT_START = 21.0
NIGHT_END = 5.0
DUSK_HOURS = 1.5
REST_DARK = 0.6  # Sheep lie down when it's darker than this...
NIGHT_HUNGER = 0.85  # ...unless they're hungrier than this.
NIGHT_HUNGER_RATE = 0.3  # Hunger rises this much slower while resting.
NIGHT_COHESION = 2.5  # Cohesion is this much stronger in the dark: the herd bunches up.
DUSK_LONELY = 1.5  # LONELY_RADIUS grows by this fraction in the dark.

# The dog. Sheep know it, so they step out of its way rather than bolt, and
# only get a little nervous, more so when it sprints.
DOG_WALK_SPEED = 25
DOG_SPRINT_SPEED = 170
DOG_SPRINT_EVERY = 35  # Mean seconds between sprints.
DOG_SPRINT_TIME = (1.0, 2.5)  # Range of sprint lengths in seconds.
DOG_ACCEL = 3.0  # How quickly the dog changes speed (per second).
DOG_TURN = 40  # Degrees/second the dog turns towards where it's heading.
DOG_WEAVE = 25  # Degrees either side of the straight line it weaves as it walks.
DOG_TRIP = (250, 900)  # Distance range to the next spot the dog wanders off to.
DOG_SNIFF_CHANCE = 0.4  # Chance it stops to sniff about when it gets there...
DOG_SNIFF_TIME = (2.0, 6.0)  # ...for this many seconds.
DOG_RADIUS = 30  # Size of the dog, for bales and for clicking it.
DOG_AVOID_RADIUS = (100, 200)  # Sheep keep this far from a walking / sprinting dog.
DOG_AVOID_FORCE = 250  # Push away from the dog, fading to zero at the avoid radius.
DOG_ALARM = (0.08, 0.45)  # Alarm close to a walking / sprinting dog.

# Colors.
GRASS = (98, 146, 60)
GRASS_DARK = (78, 124, 46)
GRASS_LIGHT = (124, 168, 76)
DIRT = (138, 110, 66)
DIRT_ALPHA = 175  # How brown a completely bare cell gets.
FLOWER_COLORS = [(240, 236, 220), (236, 210, 80)]
SHADOW_COLOR = (66, 104, 42)
WOOL_COLOR = (240, 236, 224)
WOOL_SHADE = (196, 190, 176)
THIN_WOOL = (206, 190, 176)  # A starved sheep's fleece, thin enough to show skin.
THIN_SHADE = (170, 150, 136)
FACE_COLOR = (44, 38, 36)
FENCE_COLOR = (120, 88, 56)
POST_COLOR = (84, 60, 38)
GATE_COLOR = (150, 112, 70)
HAY_COLOR = (218, 184, 96)
HAY_DARK = (172, 136, 60)
DOG_BLACK = (28, 26, 28)
DOG_WHITE = (236, 234, 228)
HUNGER_BACK = (40, 52, 30)
STATUS_TEXT = (240, 240, 230)
STATUS_GOAL = (255, 220, 120)
STATUS_BACK = (20, 30, 15, 150)
FULL_COLOR = (120, 200, 90)
HUNGRY_COLOR = (220, 70, 50)
BUBBLE_COLOR = (250, 250, 245)
BUBBLE_TEXT = (40, 36, 34)
NIGHT_TINT = (10, 16, 48)
DUSK_TINT = (255, 120, 50)
NIGHT_ALPHA = 165  # How dark the middle of the night gets.

_VIEW_COS = math.cos(math.radians(VIEW_ANGLE / 2))
_FIELD_MIN = FENCE_INSET + 12  # Sheep centres stay this far inside the window...
_FIELD_MAX_X = SCREEN_RES_X - _FIELD_MIN  # ...so the wool never pokes through the fence.
_FIELD_MAX_Y = SCREEN_RES_Y - _FIELD_MIN
_PEN_LEFT = SCREEN_RES_X - FENCE_INSET - PEN_W
_PEN_TOP = SCREEN_RES_Y - FENCE_INSET - PEN_H
_GATE_TOP = _PEN_TOP + (PEN_H - GATE_W) / 2
_GATE_BOTTOM = _GATE_TOP + GATE_W
_GATE_MID = (_GATE_TOP + _GATE_BOTTOM) / 2
_GATE_OUT = Vector2(_PEN_LEFT - 45, _GATE_MID)  # Waypoints just outside and inside the gate.
_GATE_IN = Vector2(_PEN_LEFT + 45, _GATE_MID)


def limit(v: Vector2, max_length: float) -> Vector2:
    """Return v shortened to at most max_length."""
    if v.length_squared() > max_length * max_length:
        v.scale_to_length(max_length)
    return v


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp_color(a: tuple, b: tuple, t: float) -> list[int]:
    return [round(lerp(x, y, t)) for x, y in zip(a, b)]


def smoothstep(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def turn_towards(heading: Vector2, direction: Vector2, max_degrees: float) -> None:
    """Rotate heading in place towards direction by at most max_degrees."""
    angle = (heading.angle_to(direction) + 180) % 360 - 180
    heading.rotate_ip(max(-max_degrees, min(max_degrees, angle)))


def in_pen(pos: Vector2) -> bool:
    return pos.x > _PEN_LEFT and pos.y > _PEN_TOP


def pen_walls(gate_open: bool) -> list[tuple[Vector2, Vector2]]:
    """The pen's two inner sides as line segments; the field fence is the other two."""
    right, bottom = SCREEN_RES_X - FENCE_INSET, SCREEN_RES_Y - FENCE_INSET
    walls = [(Vector2(_PEN_LEFT, _PEN_TOP), Vector2(right, _PEN_TOP)),
             (Vector2(_PEN_LEFT, _PEN_TOP), Vector2(_PEN_LEFT, _GATE_TOP)),
             (Vector2(_PEN_LEFT, _GATE_BOTTOM), Vector2(_PEN_LEFT, bottom))]
    if not gate_open:
        walls.append((Vector2(_PEN_LEFT, _GATE_TOP), Vector2(_PEN_LEFT, _GATE_BOTTOM)))
    return walls


def closest_on_wall(pos: Vector2, wall: tuple[Vector2, Vector2]) -> Vector2:
    a, b = wall
    ab = b - a
    t = min(1.0, max(0.0, (pos - a).dot(ab) / ab.length_squared()))
    return a + ab * t


def wall_distance(pos: Vector2, walls: list[tuple[Vector2, Vector2]]) -> float:
    return min((pos.distance_to(closest_on_wall(pos, w)) for w in walls), default=math.inf)


def push_out_of_walls(pos: Vector2, vel: Vector2 | None, radius: float,
                      walls: list[tuple[Vector2, Vector2]]) -> Vector2:
    """Return pos moved out of any wall it overlaps, cancelling vel into the wall."""
    for w in walls:
        c = closest_on_wall(pos, w)
        offset = pos - c
        dist = offset.length()
        if dist < radius:
            normal = offset / dist if dist > 0 else Vector2(-1, 0)
            pos = c + normal * radius
            if vel is not None:
                inward = vel.dot(normal)
                if inward < 0:
                    vel -= normal * inward
    return pos


def route(pos: Vector2, target: Vector2, gate_open: bool) -> Vector2 | None:
    """Where to head for to get to target: the target itself, or the gate if
    it's on the other side of the pen fence. None if the gate is shut."""
    here, there = in_pen(pos), in_pen(target)
    if here == there:
        return target
    if not gate_open:
        return None
    in_gateway = abs(pos.y - _GATE_MID) < GATE_W / 2 - 12 and abs(pos.x - _PEN_LEFT) < 55
    if in_gateway:
        return _GATE_IN if there else _GATE_OUT
    return _GATE_IN if here else _GATE_OUT


def darkness_at(hour: float) -> float:
    """0 in daylight, 1 at night, smooth through dusk and dawn."""
    if NIGHT_END + DUSK_HOURS <= hour < NIGHT_START - DUSK_HOURS:
        return 0.0
    if NIGHT_START - DUSK_HOURS <= hour < NIGHT_START:
        return smoothstep((hour - (NIGHT_START - DUSK_HOURS)) / DUSK_HOURS)
    if NIGHT_END <= hour < NIGHT_END + DUSK_HOURS:
        return smoothstep(1 - (hour - NIGHT_END) / DUSK_HOURS)
    return 1.0


_BUBBLES: dict[str, pygame.Surface] = {}


def bubble(text: str) -> pygame.Surface:
    """A speech bubble with text in it, made once per text."""
    if text not in _BUBBLES:
        label = pygame.font.Font(None, 20).render(text, True, BUBBLE_TEXT)
        w, h = label.get_width() + 10, label.get_height() + 6
        surf = pygame.Surface((w, h + 6), pygame.SRCALPHA)
        pygame.draw.rect(surf, BUBBLE_COLOR, (0, 0, w, h), border_radius=6)
        pygame.draw.polygon(surf, BUBBLE_COLOR, [(5, h - 1), (13, h - 1), (3, h + 6)])
        surf.blit(label, (5, 3))
        _BUBBLES[text] = surf
    return _BUBBLES[text]


class Grid:
    """Buckets sheep into VIEW_RADIUS-sized cells for fast neighbour lookup.

    A sheep can only see others within VIEW_RADIUS, so they all lie in its own
    cell or the 8 around it. This replaces checking every pair of sheep.
    """

    def __init__(self, sheep: list["Sheep"]):
        self.cells: dict[tuple[int, int], list[Sheep]] = {}
        for s in sheep:
            self.cells.setdefault(self._key(s.pos), []).append(s)

    @staticmethod
    def _key(pos: Vector2) -> tuple[int, int]:
        return int(pos.x // VIEW_RADIUS), int(pos.y // VIEW_RADIUS)

    def nearby(self, pos: Vector2, cells: int = 1) -> list["Sheep"]:
        """Sheep in the block of cells reaching `cells` cells out around pos
        (the default 3x3 block covers VIEW_RADIUS)."""
        cx, cy = self._key(pos)
        found = []
        for x in range(cx - cells, cx + cells + 1):
            for y in range(cy - cells, cy + cells + 1):
                found.extend(self.cells.get((x, y), ()))
        return found


class Grass:
    """How much grass is left on each GRASS_CELL square of the field.

    Worn cells are drawn as a brown overlay: one pixel per cell on a small
    surface, smooth-scaled up to the window so bare patches have soft edges.
    """

    def __init__(self):
        self.cols = SCREEN_RES_X // GRASS_CELL + 1
        self.rows = SCREEN_RES_Y // GRASS_CELL + 1
        self.amount = [1.0] * (self.cols * self.rows)
        self.worn: set[int] = set()  # Cells with less than full grass, to regrow.
        self.dirty: set[int] = set()  # Cells whose look changed since the last redraw.
        self.wear = pygame.Surface((self.cols, self.rows), pygame.SRCALPHA)
        self.scaled: pygame.Surface | None = None
        self._tick = 0.0
        self._since_redraw = 0.0

    def index(self, pos: Vector2) -> int:
        x = min(self.cols - 1, max(0, int(pos.x // GRASS_CELL)))
        y = min(self.rows - 1, max(0, int(pos.y // GRASS_CELL)))
        return y * self.cols + x

    def at(self, pos: Vector2) -> float:
        return self.amount[self.index(pos)]

    def eat(self, i: int, bite: float) -> None:
        self.amount[i] = max(0.0, self.amount[i] - bite)
        self.worn.add(i)
        self.dirty.add(i)

    def mean(self) -> float:
        return sum(self.amount) / len(self.amount)

    def update(self) -> None:
        self._tick += DT
        if self._tick >= GRASS_TICK:
            self._tick = 0.0
            for i in list(self.worn):
                self.amount[i] = min(1.0, self.amount[i] + GRASS_REGROW * GRASS_TICK)
                if self.amount[i] >= 1.0:
                    self.worn.discard(i)
                self.dirty.add(i)
        self._since_redraw += DT
        if self.dirty and self._since_redraw >= 0.25:
            self._since_redraw = 0.0
            for i in self.dirty:
                alpha = round((1 - self.amount[i]) * DIRT_ALPHA)
                self.wear.set_at((i % self.cols, i // self.cols), DIRT + (alpha,))
            self.dirty.clear()
            size = (self.cols * GRASS_CELL, self.rows * GRASS_CELL)
            self.scaled = pygame.transform.smoothscale(self.wear, size)

    def draw(self, surface: pygame.Surface) -> None:
        if self.scaled is not None:
            surface.blit(self.scaled, (0, 0))


class Bale:
    """A round hay bale that shrinks as sheep eat it and is gone when empty."""

    def __init__(self, pos: Vector2):
        self.pos = pos
        self.hay = 1.0  # Fraction left.
        self._spots: list[Vector2] = []
        self._spots_grid: Grid | None = None  # The grid _spots were worked out for.

    def free_spots(self, world: "World") -> list[Vector2]:
        """Places round the bale where a sheep could stand and eat, not taken.

        Worked out once per step: the grid is rebuilt every step, so a new
        grid means the old answer is stale. Places on the other side of a pen
        wall, or too close to one to stand in, don't count.
        """
        grid = world.grid
        if self._spots_grid is not grid:
            self._spots_grid = grid
            r = self.radius + SHEEP_RADIUS + HAY_MARGIN
            ring = [self.pos + Vector2(r, 0).rotate(i * 360 / RING_SPOTS) for i in range(RING_SPOTS)]
            side = in_pen(self.pos)
            self._spots = [p for p in ring
                           if in_pen(p) == side
                           and _FIELD_MIN <= p.x <= _FIELD_MAX_X and _FIELD_MIN <= p.y <= _FIELD_MAX_Y
                           and wall_distance(p, world.walls) > SHEEP_RADIUS
                           and all(p.distance_squared_to(s.pos) > SPOT_CLEAR * SPOT_CLEAR
                                   for s in grid.nearby(p))]
        return self._spots

    @property
    def radius(self) -> float:
        # Seen from above, the area shrinks with the hay, so the radius with its root.
        return lerp(HAY_MIN_RADIUS, HAY_RADIUS, math.sqrt(max(0.0, self.hay)))

    @property
    def reach(self) -> float:
        """Sheep closer than this to the centre can eat.

        Must lie beyond avoid_hay()'s push zone, or it holds sheep just out of
        reach: with a 10 px zone and a fixed 40 px reach only 4-7 sheep ate.
        """
        return self.radius + SHEEP_RADIUS + HAY_MARGIN + EAT_GAP

    def draw(self, surface: pygame.Surface) -> None:
        draw_hay(surface, self.pos, self.radius)


class Sheep:
    """A sheep (or lamb) with a position, velocity and facing, steered by
    Reynolds-style forces.

    Unlike a bird a sheep can stand still, so the way it faces is kept
    separately from its velocity and turns towards it only while moving.
    """

    def __init__(self, pos: Vector2, mother: "Sheep | None" = None):
        self.pos = pos
        self.mother = mother  # Set for lambs.
        self.size = LAMB_SIZE if mother is not None else 1.0
        self.radius = SHEEP_RADIUS * self.size
        self.heading = Vector2(1, 0).rotate(random.uniform(0, 360))
        self.vel = self.heading * GRAZE_SPEED
        self.walking = False
        self.alarm = 0.0  # 0 = calm, 1 = running for its life.
        self.wander = 0.0  # Smoothly varying sideways drift, roughly in [-1, 1].
        self.chew = random.uniform(0, math.tau)  # Head-bob phase, only for drawing.
        self.hunger = random.random()
        self.full = False  # Set when hay brings hunger to 0, cleared at HUNGRY_AT.
        self.grass_eating = False  # Between GRASS_HUNGER and GRASS_SATED, see seek_grass().
        self.wool = 1.0  # 1 = full fleece; starving thins it.
        self.resting = False  # Lying down for the night.
        self.rest_pose = 0.0  # 0 standing .. 1 lying, eased, only for drawing.
        self.following = False  # A lamb going after its mother.
        self.inside = in_pen(pos)
        self.bleat_text = ""
        self.bleat_left = 0.0
        self._grass_target: Vector2 | None = None
        # State changes decided in acceleration() and applied in move(), so
        # every sheep in a step reacts to the same snapshot of the herd.
        self._next_walking = False
        self._next_alarm = 0.0
        self._face: Vector2 | None = None  # Point to turn towards: the bale while eating or queueing, away when full.
        self._eating: Bale | None = None  # The bale it's eating from this step.
        self._grazing: int | None = None  # The grass cell it's eating from this step.

    def say(self, text: str) -> None:
        self.bleat_text = text
        self.bleat_left = BLEAT_TIME

    def neighbours(self, grid: Grid) -> list[tuple["Sheep", Vector2, float]]:
        """Visible sheep within VIEW_RADIUS, as (sheep, offset to it, distance).

        Sheep on the other side of the pen fence don't count, or the herd
        outside would hold the ones inside pressed against it.
        """
        found = []
        for s in grid.nearby(self.pos):
            if s is self or s.inside != self.inside:
                continue
            offset = s.pos - self.pos
            dist_sq = offset.length_squared()
            if dist_sq >= VIEW_RADIUS * VIEW_RADIUS or dist_sq == 0:
                continue
            dist = math.sqrt(dist_sq)
            # Skip sheep in the blind spot behind us.
            if self.heading.dot(offset) / dist < _VIEW_COS:
                continue
            found.append((s, offset, dist))
        return found

    def flocking(self, near: list[tuple["Sheep", Vector2, float]], darkness: float) -> Vector2:
        """Separation, alignment and cohesion combined into one force."""
        if not near:
            return Vector2()

        away = Vector2()
        mean_vel = Vector2()
        centre = Vector2()
        for s, offset, dist in near:
            if dist < SEPARATION_RADIUS:
                # Push away harder the closer the neighbour is.
                away -= offset / dist * (1 - dist / SEPARATION_RADIUS)
            mean_vel += s.vel
            centre += offset

        force = away * SEPARATION_FORCE
        force += (mean_vel / len(near) - self.vel) * ALIGNMENT_GAIN
        cohesion = COHESION_GAIN * (1 + (ALARM_COHESION - 1) * self.alarm)
        cohesion *= 1 + (NIGHT_COHESION - 1) * darkness
        if self._next_walking:
            cohesion *= WALK_COHESION
        force += centre / len(near) * cohesion
        return force

    def update_state(self, near: list[tuple["Sheep", Vector2, float]]) -> None:
        """Decide whether to graze or walk next step, and how alarmed to be."""
        # Fear is contagious: take on most of the most frightened neighbour's alarm.
        alarm = self.alarm * math.exp(-DT / ALARM_TIME)
        for s, _, _ in near:
            alarm = max(alarm, s.alarm * ALARM_SPREAD)
        self._next_alarm = alarm

        if self.walking:
            self._next_walking = random.random() >= DT / WALK_TIME
        else:
            # Sheep follow each other: walking neighbours make it more likely to set off.
            walkers = sum(1 for s, _, _ in near if s.walking) / len(near) if near else 0
            self._next_walking = random.random() < DT / GRAZE_TIME * (1 + FOLLOW * walkers)

    def steer(self, target: Vector2, speed: float) -> Vector2:
        """Force that walks the sheep towards target at speed."""
        to = target - self.pos
        if to.length_squared() < 1:
            return Vector2()
        return (to.normalize() * speed - self.vel) * SPEED_GAIN

    def edges(self) -> Vector2:
        """Turn back from the fence, harder the closer the sheep gets."""
        force = Vector2()
        if self.pos.x < _FIELD_MIN + EDGE_MARGIN:
            force.x = 1 - (self.pos.x - _FIELD_MIN) / EDGE_MARGIN
        elif self.pos.x > _FIELD_MAX_X - EDGE_MARGIN:
            force.x = -(1 - (_FIELD_MAX_X - self.pos.x) / EDGE_MARGIN)
        if self.pos.y < _FIELD_MIN + EDGE_MARGIN:
            force.y = 1 - (self.pos.y - _FIELD_MIN) / EDGE_MARGIN
        elif self.pos.y > _FIELD_MAX_Y - EDGE_MARGIN:
            force.y = -(1 - (_FIELD_MAX_Y - self.pos.y) / EDGE_MARGIN)
        return force * EDGE_FORCE

    def avoid_walls(self, walls: list[tuple[Vector2, Vector2]]) -> Vector2:
        """Push away from the pen walls just before touching them; move() makes them solid."""
        force = Vector2()
        for w in walls:
            offset = self.pos - closest_on_wall(self.pos, w)
            dist = offset.length()
            if 0 < dist < self.radius + WALL_MARGIN:
                force += offset / dist * WALL_PUSH * min(1.0, 1 - (dist - self.radius) / WALL_MARGIN)
        return force

    def avoid_hay(self, bales: list[Bale]) -> Vector2:
        """Push away from bales just before touching them.

        This only keeps the crowd from pressing sheep into a bale; move()
        is what makes bales solid.
        """
        force = Vector2()
        for b in bales:
            contact = b.radius + self.radius
            offset = self.pos - b.pos
            dist = offset.length()
            if 0 < dist < contact + HAY_MARGIN:
                force += offset / dist * HAY_PUSH * min(1.0, 1 - (dist - contact) / HAY_MARGIN)
        return force

    def follow_mother(self, world: "World") -> Vector2:
        """A lamb trots after its mother once she's more than LAMB_STAY away,
        until it's back within LAMB_CLOSE; it bleats if it has lost her."""
        to = self.mother.pos - self.pos
        dist = to.length()
        self.following = dist > (LAMB_CLOSE if self.following else LAMB_STAY)
        if not self.following:
            return Vector2()
        if dist > LAMB_LOST and random.random() < DT * LAMB_BLEAT_RATE:
            self.say("Maa!")
        target = route(self.pos, self.mother.pos, world.gate_open)
        if target is None:  # Shut out of the pen: all it can do is call.
            self.following = False
            return Vector2()
        self._next_walking = True
        return self.steer(target, LAMB_TROT)

    def seek_food(self, world: "World", near: list[tuple["Sheep", Vector2, float]]) -> Vector2:
        """Walk towards the nearest hay bale the sheep can see.

        The hungrier the sheep, the further it looks (FOOD_RADIUS stretched by
        HUNGER_SIGHT); with a fixed radius, sheep that started far from the
        hay stood starving for minutes. Sheep further away don't notice it
        themselves, but alignment and cohesion still let them follow the
        others that do.

        Hungry sheep head for the nearest free place round the bale, walking
        round to the far side if they have to. Only when every place is taken
        does a sheep blocked by a standing sheep in front of it stop and graze
        where it is; if it kept pushing, the crowd behind crushed into a pile
        around the bale. Queueing behind the first sheep in the way, with no
        free places, left only 3-4 eating per bale, all on the herd's side.
        """
        bales = [b for b in world.bales if world.gate_open or in_pen(b.pos) == self.inside]
        if not bales:
            return Vector2()
        bale = min(bales, key=lambda b: self.pos.distance_squared_to(b.pos))
        food = bale.pos
        to_food = food - self.pos
        dist = to_food.length()
        if self.full:
            if dist >= LEAVE_DISTANCE or dist == 0:
                return Vector2()
            # Turn round and walk off, making room for the hungry ones.
            self._next_walking = True
            self._face = self.pos - to_food
            return (-to_food / dist * WALK_SPEED - self.vel) * SPEED_GAIN
        sight = FOOD_RADIUS * (1 + HUNGER_SIGHT * self.hunger)
        if dist >= sight:
            return Vector2()
        if dist < bale.reach:
            self._next_walking = False  # Arrived: stop and eat.
            self._eating = bale
            self._face = food
            return Vector2()
        # Fade in at the edge so sheep don't jerk as they cross the radius.
        strength = min(1.0, (sight - dist) / FOOD_FADE) * FOOD_WEIGHT
        spots = bale.free_spots(world)
        if spots:
            spot = min(spots, key=self.pos.distance_squared_to)
            target = route(self.pos, spot, world.gate_open)
            if target is not None and target.distance_squared_to(self.pos) > 1:
                self._next_walking = True
                return self.steer(target, WALK_SPEED) * strength
        for s, offset, d in near:
            if (d < SEPARATION_RADIUS and s.vel.length() < BLOCKED_SPEED
                    and offset.dot(to_food) > QUEUE_COS * d * dist):
                self._next_walking = False  # Queueing behind another sheep.
                self._face = food
                return Vector2()
        target = route(self.pos, food, world.gate_open)
        if target is None:
            return Vector2()
        self._next_walking = True
        return self.steer(target, WALK_SPEED) * strength

    def seek_grass(self, world: "World") -> Vector2:
        """Eat grass if hungry and there's no hay about, or walk to greener grass.

        Sheep start at GRASS_HUNGER and keep eating down to GRASS_SATED; the
        gap stops them flicking between eating and not at one threshold. A
        sheep on a bare cell looks round at GRASS_LOOK distances for the
        greenest cell on its side of the pen fence and walks there, which is
        what moves the herd round the field over time.
        """
        self.grass_eating = self.hunger > (GRASS_SATED if self.grass_eating else GRASS_HUNGER)
        if not self.grass_eating:
            self._grass_target = None
            return Vector2()
        grass = world.grass
        here = grass.at(self.pos)
        if here > GRASS_MIN:
            self._grass_target = None
            self._grazing = grass.index(self.pos)
            self._next_walking = False
            return Vector2()
        target = self._grass_target
        if target is None or grass.at(target) <= here + 0.1 or target.distance_to(self.pos) < 10:
            target, best = None, here + 0.1
            offset = random.uniform(0, 45)  # Don't always look the same ways.
            for dist in GRASS_LOOK:
                for i in range(8):
                    p = self.pos + Vector2(dist, 0).rotate(offset + i * 45)
                    if not (_FIELD_MIN <= p.x <= _FIELD_MAX_X and _FIELD_MIN <= p.y <= _FIELD_MAX_Y):
                        continue
                    if in_pen(p) != self.inside:
                        continue
                    amount = grass.at(p)
                    if amount > best:
                        target, best = p, amount
            self._grass_target = target
        if target is None:
            return Vector2()
        self._next_walking = True
        return self.steer(target, WALK_SPEED)

    def seek_company(self, world: "World") -> Vector2:
        """Walk back to the nearest sheep if there's none within VIEW_RADIUS.

        Without this the herd slowly fell apart: after 10 minutes it was in
        15-23 groups with lone sheep dotted round the field. At dusk the
        search reaches further, so the herd gathers for the night.
        """
        reach = LONELY_RADIUS * (1 + DUSK_LONELY * world.darkness)
        nearest, best = None, reach * reach
        for s in world.grid.nearby(self.pos, math.ceil(reach / VIEW_RADIUS)):
            if s is self:
                continue
            d = self.pos.distance_squared_to(s.pos)
            if d < VIEW_RADIUS * VIEW_RADIUS:
                return Vector2()  # Not alone after all, just not looking at them.
            if d < best and (world.gate_open or s.inside == self.inside):
                nearest, best = s, d
        if nearest is None:
            return Vector2()
        target = route(self.pos, nearest.pos, world.gate_open)
        self._next_walking = True
        return self.steer(target, WALK_SPEED)

    def avoid_dog(self, dog: "Dog | None") -> Vector2:
        """Keep out of the dog's way, fading to zero at its avoid radius.

        It must weaken with distance; a repulsion that grows with distance
        overpowers edges() and pins sheep against the fence. A sprinting dog
        is given more room and makes sheep more nervous than a walking one.
        """
        if dog is None or in_pen(dog.pos) != self.inside:
            return Vector2()
        away = self.pos - dog.pos
        dist = away.length()
        radius = lerp(*DOG_AVOID_RADIUS, dog.sprint)
        if dist >= radius:
            return Vector2()
        closeness = 1 - dist / radius
        # Full alarm within the inner half of the radius.
        alarm = lerp(*DOG_ALARM, dog.sprint) * min(1.0, 2 * closeness)
        self._next_alarm = max(self._next_alarm, alarm)
        if dist == 0:
            away = Vector2(self.heading)
        return away.normalize() * DOG_AVOID_FORCE * closeness

    def strength(self) -> float:
        """1 normally, down to 1 - STARVE_SLOW when completely starved."""
        return 1 - STARVE_SLOW * max(0.0, (self.hunger - STARVING) / (1 - STARVING))

    def acceleration(self, world: "World") -> Vector2:
        """Total steering force for this step.

        What a calm sheep does, in order of preference: a lamb that's too far
        from its mother goes after her; at night it lies down; otherwise hay,
        then grass if hungry enough, then company if it's alone.
        """
        near = self.neighbours(world.grid)
        self.update_state(near)
        self._face = None
        self._eating = None
        self._grazing = None
        force = self.flocking(near, world.darkness)

        task = Vector2()
        if self.mother is not None:
            task = self.follow_mother(world)
        self.resting = (not self.following and world.darkness > REST_DARK
                        and self.hunger < NIGHT_HUNGER and self.alarm < 0.2)
        if self.resting:
            self._next_walking = False
        elif not self.following:
            task = self.seek_food(world, near)
            if (self._face is not None and self._eating is None and not self.full
                    and self.hunger > GRASS_HUNGER and world.grass.at(self.pos) > GRASS_MIN):
                # Queueing for hay: nibble the grass underfoot while waiting.
                # Without this, sheep stuck in a queue starved standing on grass.
                self._grazing = world.grass.index(self.pos)
            elif task.length_squared() == 0 and self._face is None:
                task = self.seek_grass(world)
                if task.length_squared() == 0 and self._grazing is None and not near:
                    task = self.seek_company(world)
        force += task

        if self.following:
            calm_speed = LAMB_TROT
        elif self._next_walking:
            calm_speed = WALK_SPEED
        else:
            calm_speed = 0 if self.resting else GRAZE_SPEED
        goal_speed = lerp(calm_speed, RUN_SPEED, self.alarm) * self.strength()
        # Smooth random walk (Ornstein-Uhlenbeck), not per-frame noise, so
        # sheep meander instead of twitching. The sideways push scales with
        # speed: at full strength on a grazing sheep it outran GRAZE_SPEED,
        # the body turned to follow it, and sheep walked in endless circles.
        # Grazing sheep instead turn slowly on the spot, in move().
        decay = math.exp(-DT / WANDER_TIME)
        self.wander = self.wander * decay + random.gauss(0, math.sqrt(1 - decay * decay))
        force += self.heading.rotate(90) * self.wander * WANDER_FORCE * min(1.0, goal_speed / WALK_SPEED)
        force = limit(force, lerp(MAX_FORCE, PANIC_FORCE, self.alarm))
        # Added after the limit so flocking can't override them. Sheep have
        # footing, not momentum: this pulls the whole velocity towards the goal,
        # which is what lets them stop and stand still to graze.
        force += (self.heading * goal_speed - self.vel) * SPEED_GAIN
        return (force + self.edges() + self.avoid_walls(world.walls)
                + self.avoid_hay(world.bales) + self.avoid_dog(world.dog))

    def feed(self, world: "World") -> None:
        """Hunger, wool and bleating for this step."""
        rate = HUNGER_RATE * (NIGHT_HUNGER_RATE if self.resting else 1)
        if self._eating is not None:
            self._eating.hay -= DT / BALE_HAY
            self.hunger = max(0.0, self.hunger - EAT_RATE * DT)
            if self.hunger == 0:
                self.full = True
        else:
            if self._grazing is not None and self.vel.length() < 10:
                world.grass.eat(self._grazing, GRASS_BITE * DT)
                rate -= GRASS_EAT_RATE
            self.hunger = min(1.0, max(0.0, self.hunger + rate * DT))
            if self.full and self.hunger >= HUNGRY_AT:
                self.full = False

        if self.hunger >= STARVING:
            self.wool = max(WOOL_MIN, self.wool - WOOL_LOSS * DT)
            if random.random() < DT * BLEAT_RATE:
                self.say("Baa!")
        elif self.hunger < WOOL_REGROW_BELOW:
            self.wool = min(1.0, self.wool + WOOL_REGROW * DT)
        self.bleat_left = max(0.0, self.bleat_left - DT)

    def move(self, accel: Vector2, world: "World") -> None:
        self.walking = self._next_walking
        self.alarm = self._next_alarm
        self.feed(world)
        self.vel += accel * DT
        limit(self.vel, lerp(MAX_SPEED, PANIC_SPEED, self.alarm))
        self.pos += self.vel * DT

        # The fence is solid: stop at it instead of walking through.
        if not _FIELD_MIN <= self.pos.x <= _FIELD_MAX_X:
            self.pos.x = min(max(self.pos.x, _FIELD_MIN), _FIELD_MAX_X)
            self.vel.x = 0
        if not _FIELD_MIN <= self.pos.y <= _FIELD_MAX_Y:
            self.pos.y = min(max(self.pos.y, _FIELD_MIN), _FIELD_MAX_Y)
            self.vel.y = 0
        self.pos = push_out_of_walls(self.pos, self.vel, self.radius, world.walls)

        # Bales are solid too: slide round them instead of standing on them.
        for b in world.bales:
            contact = b.radius + self.radius
            offset = self.pos - b.pos
            dist = offset.length()
            if dist < contact:
                normal = offset / dist if dist > 0 else -self.heading
                self.pos = b.pos + normal * contact
                inward = self.vel.dot(normal)
                if inward < 0:
                    self.vel -= normal * inward
        self.inside = in_pen(self.pos)

        # Turn the body towards where we're going, at a limited rate. Slow
        # nudges from neighbours are just side-steps: if the body followed
        # them, sheep in a tight group would spin on the spot.
        speed = self.vel.length()
        facing = min(1.0, max(0.0, (speed - FACE_SPEED) / (FACE_FULL_SPEED - FACE_SPEED)))
        if facing > 0:
            turn_towards(self.heading, self.vel, TURN_RATE * DT * facing)
        if self._face is not None and self.alarm < 0.1:
            # Stopped at a bale: face the hay instead of drifting round, give
            # or take a little sway so the crowd doesn't line up like soldiers.
            # Full sheep use this to turn their back on it before walking off.
            target = (self._face - self.pos).rotate(self.wander * HAY_SWAY)
            turn_towards(self.heading, target, HAY_TURN * DT * (1 - facing))
        elif not self.walking and not self.resting:
            self.heading.rotate_ip(self.wander * GRAZE_TURN * DT * (1 - facing))
        lying = 1.0 if self.resting and speed < 5 else 0.0
        self.rest_pose += (lying - self.rest_pose) * min(1.0, 2 * DT)
        self.chew += DT * (3 + 6 * speed / RUN_SPEED)

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the sheep from above: a woolly body with a dark face.

        Lambs are smaller, starving sheep have a thin greyish fleece, and a
        sheep lying down is rounder with its head turned along its side.
        """
        fwd = self.heading
        side = fwd.rotate(90)
        p = self.pos
        sc = self.size
        rest = self.rest_pose
        speed = self.vel.length()
        # Head is lowered and nods side to side while grazing, up while walking.
        grazing = max(0.0, 1 - speed / WALK_SPEED) * (1 - rest)
        head = (p + fwd * lerp(12 - 2 * grazing, 8, rest) * sc
                + side * (1.5 * grazing * math.sin(self.chew) + 5 * rest) * sc)

        body = lerp(1.0, 0.65, rest) * sc  # Lying down, the body is rounder.
        puff = sc * lerp(0.75, 1.0, self.wool) * lerp(1.0, 1.08, rest)
        fleece = [(p - fwd * 5 * body, 7.5), (p + fwd * 2 * body, 8),
                  (p - fwd * 1 * body + side * 3.5 * body, 6.5),
                  (p - fwd * 1 * body - side * 3.5 * body, 6.5)]
        thin = 1 - (self.wool - WOOL_MIN) / (1 - WOOL_MIN)
        wool = lerp_color(WOOL_COLOR, THIN_WOOL, thin)
        shade = lerp_color(WOOL_SHADE, THIN_SHADE, thin)
        for c, r in fleece:
            pygame.draw.circle(surface, SHADOW_COLOR, c + Vector2(3, 4) * sc, r * puff)
        for c, r in fleece:
            pygame.draw.circle(surface, shade, c, r * puff + 1)
        for c, r in fleece:
            pygame.draw.circle(surface, wool, c, r * puff)

        hs = sc ** 0.7  # Lambs' heads are big for their bodies.
        for s in (1, -1):  # Ears.
            pygame.draw.circle(surface, FACE_COLOR, head - fwd * 1.5 * hs + side * 4.5 * s * hs, 2.2 * hs)
        pygame.draw.circle(surface, FACE_COLOR, head, 4 * hs)
        pygame.draw.circle(surface, FACE_COLOR, head + fwd * 2.5 * hs, 3 * hs)

    def draw_bubble(self, surface: pygame.Surface) -> None:
        if self.bleat_left > 0:
            b = bubble(self.bleat_text)
            surface.blit(b, (self.pos.x + 4, self.pos.y - 14 * self.size - b.get_height()))

    def draw_hunger(self, surface: pygame.Surface) -> None:
        """A small bar above the sheep showing how full it is: full and green
        when it's had enough, short and red when it's hungry."""
        fullness = 1 - self.hunger
        x, y = self.pos.x - 10, self.pos.y - 20 * self.size
        pygame.draw.rect(surface, HUNGER_BACK, (x, y, 20, 4))
        color = lerp_color(HUNGRY_COLOR, FULL_COLOR, fullness)
        pygame.draw.rect(surface, color, (x, y, round(20 * fullness), 4))


class Dog:
    """A sheepdog that wanders round the field on its own, with the odd sprint.

    It picks a spot somewhere in the field and ambles over to it, weaving a
    little, sometimes stops there to sniff about, then picks the next one.
    Steering straight from the random walk instead (turning at wander * rate)
    made it walk in tight circles whenever the wander stayed one way. It
    keeps out of the pen unless it's put there, and sleeps at night.
    """

    def __init__(self, pos: Vector2, world: "World"):
        self.pos = pos
        self.heading = Vector2(1, 0).rotate(random.uniform(0, 360))
        self.speed = DOG_WALK_SPEED
        self.sprint = 0.0  # 0 = walking, 1 = at full sprint; how scary it is.
        self.sprint_left = 0.0  # Seconds of the current sprint still to run.
        self.sniff_left = 0.0  # Seconds left standing and sniffing.
        self.target = self.pick_target(world)
        self.wander = 0.0
        self.wag = 0.0

    def pick_target(self, world: "World") -> Vector2:
        """A spot well inside the fence, a walk of DOG_TRIP away if possible.

        Outside the pen, unless the dog is shut inside it.
        """
        shut_in = in_pen(self.pos) and not world.gate_open
        for _ in range(20):
            if shut_in:
                spot = Vector2(random.uniform(_PEN_LEFT + 40, _FIELD_MAX_X - 40),
                               random.uniform(_PEN_TOP + 40, _FIELD_MAX_Y - 40))
                break
            margin = EDGE_MARGIN + 100
            spot = Vector2(random.uniform(_FIELD_MIN + margin, _FIELD_MAX_X - margin),
                           random.uniform(_FIELD_MIN + margin, _FIELD_MAX_Y - margin))
            if spot.x > _PEN_LEFT - 100 and spot.y > _PEN_TOP - 100:
                continue
            if DOG_TRIP[0] <= spot.distance_to(self.pos) <= DOG_TRIP[1]:
                break
        return spot

    def update(self, world: "World") -> None:
        asleep = world.darkness > REST_DARK
        if asleep:
            self.sprint_left = 0.0
        elif self.sprint_left > 0:
            self.sprint_left -= DT
            if self.sprint_left <= 0:
                self.target = self.pick_target(world)  # Lost interest; wander on from here.
        elif random.random() < DT / DOG_SPRINT_EVERY:
            self.sprint_left = random.uniform(*DOG_SPRINT_TIME)
            self.sniff_left = 0
            self.heading.rotate_ip(random.uniform(-60, 60))  # Off after something.
        elif self.sniff_left > 0:
            self.sniff_left -= DT
        elif self.pos.distance_to(self.target) < 40:
            if random.random() < DOG_SNIFF_CHANCE:
                self.sniff_left = random.uniform(*DOG_SNIFF_TIME)
            self.target = self.pick_target(world)

        if asleep:
            goal = 0
        elif self.sprint_left > 0:
            goal = DOG_SPRINT_SPEED
        elif self.sniff_left > 0:
            goal = 0
        else:
            goal = DOG_WALK_SPEED
        self.speed += (goal - self.speed) * min(1.0, DOG_ACCEL * DT)
        self.sprint = max(0.0, (self.speed - DOG_WALK_SPEED) / (DOG_SPRINT_SPEED - DOG_WALK_SPEED))

        # The same smooth random walk the sheep use, here only bending the
        # line to the target a little, or turning the nose about while sniffing.
        decay = math.exp(-DT / WANDER_TIME)
        self.wander = self.wander * decay + random.gauss(0, math.sqrt(1 - decay * decay))
        if asleep:
            pass
        elif self.sniff_left > 0:
            self.heading.rotate_ip(self.wander * DOG_TURN * DT)
        elif self.sprint_left <= 0:
            waypoint = route(self.pos, self.target, world.gate_open)
            if waypoint is None:  # The gate shut between it and where it was going.
                self.target = self.pick_target(world)
            elif waypoint.distance_squared_to(self.pos) > 1:
                want = (waypoint - self.pos).rotate(self.wander * DOG_WEAVE)
                turn_towards(self.heading, want, DOG_TURN * DT)
        else:
            # Near the fence (only reached by sprinting), swing back towards the middle.
            margin = EDGE_MARGIN + 40
            if not (_FIELD_MIN + margin < self.pos.x < _FIELD_MAX_X - margin
                    and _FIELD_MIN + margin < self.pos.y < _FIELD_MAX_Y - margin):
                centre = Vector2(SCREEN_RES_X / 2, SCREEN_RES_Y / 2)
                turn_towards(self.heading, centre - self.pos, 150 * DT)

        self.pos += self.heading * self.speed * DT
        self.pos.x = min(max(self.pos.x, _FIELD_MIN), _FIELD_MAX_X)
        self.pos.y = min(max(self.pos.y, _FIELD_MIN), _FIELD_MAX_Y)
        self.pos = push_out_of_walls(self.pos, None, DOG_RADIUS * 0.4, world.walls)
        for b in world.bales:  # Goes round bales, not over them.
            contact = b.radius + DOG_RADIUS * 0.5
            offset = self.pos - b.pos
            dist = offset.length()
            if dist < contact:
                normal = offset / dist if dist > 0 else -self.heading
                self.pos = b.pos + normal * contact
        self.wag += DT * (1 if asleep else 6 + 12 * self.speed / DOG_SPRINT_SPEED)

    def draw(self, surface: pygame.Surface) -> None:
        draw_dog(surface, self.pos, self.heading, self.wag)


class World:
    """Everything in the field, stepped together."""

    def __init__(self):
        self.herd = make_herd()
        self.bales: list[Bale] = []
        self.dog: Dog | None = None
        self.grass = Grass()
        self.gate_open = True
        self.walls = pen_walls(self.gate_open)
        self.time = 0.0
        self.darkness = darkness_at(START_HOUR)
        self.penned_at: float | None = None  # Time the whole herd was first in the pen.
        self.grid = Grid(self.herd)

    @property
    def hour(self) -> float:
        return (START_HOUR + 24 * self.time / DAY_LENGTH) % 24

    @property
    def day(self) -> int:
        return 1 + int((START_HOUR / 24 * DAY_LENGTH + self.time) // DAY_LENGTH)

    def toggle_gate(self) -> None:
        self.gate_open = not self.gate_open
        self.walls = pen_walls(self.gate_open)

    def click_hay(self, pos: Vector2) -> None:
        hit = [b for b in self.bales if b.pos.distance_to(pos) < b.radius]
        if hit:
            self.bales.remove(hit[0])
        else:
            self.bales.append(Bale(pos))

    def click_dog(self, pos: Vector2) -> None:
        if self.dog is not None and self.dog.pos.distance_to(pos) < DOG_RADIUS:
            self.dog = None  # Called off.
        else:
            self.dog = Dog(pos, self)

    def step(self) -> None:
        self.darkness = darkness_at(self.hour)
        # All sheep steer from the same snapshot of the herd, then move.
        self.grid = Grid(self.herd)
        accels = [s.acceleration(self) for s in self.herd]
        for s, accel in zip(self.herd, accels):
            s.move(accel, self)
        self.bales = [b for b in self.bales if b.hay > 0]  # Eaten up.
        if self.dog is not None:
            self.dog.update(self)
        self.grass.update()
        self.time += DT
        if self.penned_at is None and all(s.inside for s in self.herd):
            self.penned_at = self.time

    def draw(self, screen: pygame.Surface, pasture: pygame.Surface, night: pygame.Surface,
             font: pygame.font.Font, show_hunger: bool) -> None:
        screen.blit(pasture, (0, 0))
        self.grass.draw(screen)
        draw_gate(screen, self.gate_open)
        for b in self.bales:
            b.draw(screen)
        for s in self.herd:
            s.draw(screen)
        if self.dog is not None:
            # Drawn after the sheep: the dog runs among them.
            self.dog.draw(screen)
        if self.darkness > 0:
            # Warm at dusk and dawn, deep blue at night.
            glow = 4 * self.darkness * (1 - self.darkness)
            color = lerp_color(NIGHT_TINT, DUSK_TINT, 0.6 * glow)
            night.fill(color + [round(NIGHT_ALPHA * self.darkness + 40 * glow)])
            screen.blit(night, (0, 0))
        for s in self.herd:
            s.draw_bubble(screen)
        if show_hunger:
            for s in self.herd:
                s.draw_hunger(screen)
        draw_status(screen, font, self)


def draw_hay(surface: pygame.Surface, pos: Vector2, radius: float = HAY_RADIUS) -> None:
    """A round hay bale seen from above, rolled into a spiral."""
    pygame.draw.circle(surface, SHADOW_COLOR, pos + Vector2(4, 5), radius)
    pygame.draw.circle(surface, HAY_COLOR, pos, radius)
    for f in (0.8, 0.53, 0.26):
        pygame.draw.circle(surface, HAY_DARK, pos, radius * f, 1)
    for a in range(0, 360, 40):  # Loose straws sticking out.
        d = Vector2(1, 0).rotate(a + 13)
        pygame.draw.aaline(surface, HAY_DARK, pos + d * radius * 0.9, pos + d.rotate(12) * radius * 1.26)


def draw_gate(surface: pygame.Surface, gate_open: bool) -> None:
    """The pen gate: across the gap when shut, swung into the pen when open."""
    hinge = Vector2(_PEN_LEFT, _GATE_TOP)
    tip = Vector2(_PEN_LEFT, _GATE_BOTTOM) if not gate_open else hinge + Vector2(GATE_W - 8, 6)
    pygame.draw.line(surface, SHADOW_COLOR, hinge + Vector2(3, 4), tip + Vector2(3, 4), 6)
    pygame.draw.line(surface, GATE_COLOR, hinge, tip, 6)
    pygame.draw.line(surface, POST_COLOR, hinge, tip, 2)
    for post in (hinge, Vector2(_PEN_LEFT, _GATE_BOTTOM)):
        pygame.draw.circle(surface, POST_COLOR, post, 7)


def draw_status(surface: pygame.Surface, font: pygame.font.Font, world: World) -> None:
    """Two lines in the top-left corner: how the herd is doing, and the controls."""
    herd = world.herd
    lambs = sum(s.mother is not None for s in herd)
    hungry = sum(s.hunger > 0.75 for s in herd)
    starving = sum(s.hunger >= STARVING for s in herd)
    eating = sum(s._eating is not None for s in herd)
    grazing = sum(s._grazing is not None for s in herd)
    full = sum(s.full for s in herd)
    penned = sum(s.inside for s in herd)
    hour = world.hour
    clock = f"{int(hour):02d}:{int(hour % 1 * 60):02d}"
    lines = [(f"Day {world.day}  {clock}    {len(herd) - lambs} sheep + {lambs} lambs    "
              f"hungry {hungry} (starving {starving})    eating hay {eating}    grazing {grazing}    "
              f"full {full}    bales {len(world.bales)}    grass {world.grass.mean():.0%}    "
              f"penned {penned}/{len(herd)}", STATUS_TEXT)]
    if world.penned_at is not None:
        t = round(world.penned_at)
        lines.append((f"The whole herd is in the pen! It took {t // 60}:{t % 60:02d}.", STATUS_GOAL))
    lines.append((f"left click: hay    right click: dog    G: {'close' if world.gate_open else 'open'} gate    "
                  f"H: bars    R: restart    Esc: quit", STATUS_TEXT))
    labels = [font.render(text, True, color) for text, color in lines]
    width = max(label.get_width() for label in labels) + 20
    height = sum(label.get_height() + 4 for label in labels) + 8
    panel = pygame.Surface((width, height), pygame.SRCALPHA)
    panel.fill(STATUS_BACK)
    y = 6
    for label in labels:
        panel.blit(label, (10, y))
        y += label.get_height() + 4
    surface.blit(panel, (FENCE_INSET + 14, FENCE_INSET + 12))


def draw_dog(surface: pygame.Surface, pos: Vector2, heading: Vector2, wag: float) -> None:
    """A black-and-white sheepdog seen from above, tail wagging."""
    fwd = heading
    side = fwd.rotate(90)
    tail_tip = pos - fwd * 27 + side * 7 * math.sin(wag)
    pygame.draw.line(surface, DOG_BLACK, pos - fwd * 12, tail_tip, 5)
    pygame.draw.circle(surface, DOG_WHITE, tail_tip, 3)
    body = [pos + fwd * (15 * math.cos(t)) + side * (7 * math.sin(t))
            for t in (i * math.tau / 24 for i in range(24))]
    pygame.draw.polygon(surface, DOG_BLACK, body)
    pygame.draw.circle(surface, DOG_WHITE, pos + fwd * 9, 7)  # White ruff.
    head = pos + fwd * 15
    for s in (1, -1):  # Ears.
        pygame.draw.circle(surface, DOG_BLACK, head - fwd * 2 + side * 5 * s, 3)
    pygame.draw.circle(surface, DOG_BLACK, head, 6)
    pygame.draw.circle(surface, DOG_WHITE, head + fwd * 5, 3)  # Muzzle.
    pygame.draw.circle(surface, DOG_BLACK, head + fwd * 7.5, 1.5)  # Nose.


def draw_fence(surface: pygame.Surface, a: Vector2, b: Vector2) -> None:
    """A straight run of fence with posts every 60 px."""
    pygame.draw.line(surface, SHADOW_COLOR, a + Vector2(3, 4), b + Vector2(3, 4), 4)
    pygame.draw.line(surface, FENCE_COLOR, a, b, 4)
    length = a.distance_to(b)
    for i in range(int(length // 60) + 1):
        pygame.draw.circle(surface, POST_COLOR, a.lerp(b, min(1.0, i * 60 / length)), 5)
    pygame.draw.circle(surface, POST_COLOR, b, 5)


def make_pasture() -> pygame.Surface:
    """Grass with tufts and flowers, fenced in at FENCE_INSET, with the pen."""
    rng = random.Random(7)  # Same field every run.
    field = pygame.Surface((SCREEN_RES_X, SCREEN_RES_Y))
    field.fill(GRASS)
    # Large soft patches of lusher and drier grass.
    patches = pygame.Surface((SCREEN_RES_X, SCREEN_RES_Y), pygame.SRCALPHA)
    for _ in range(40):
        color = rng.choice([GRASS_DARK, GRASS_LIGHT]) + (40,)
        pos = (rng.uniform(0, SCREEN_RES_X), rng.uniform(0, SCREEN_RES_Y))
        pygame.draw.circle(patches, color, pos, rng.uniform(80, 220))
    field.blit(patches, (0, 0))
    for _ in range(SCREEN_RES_X * SCREEN_RES_Y // 400):
        x, y = rng.uniform(0, SCREEN_RES_X), rng.uniform(0, SCREEN_RES_Y)
        color = rng.choice([GRASS_DARK, GRASS_LIGHT])
        pygame.draw.line(field, color, (x, y), (x + rng.uniform(-2, 2), y - rng.uniform(2, 5)))
    for _ in range(SCREEN_RES_X * SCREEN_RES_Y // 8000):
        pos = (rng.uniform(0, SCREEN_RES_X), rng.uniform(0, SCREEN_RES_Y))
        pygame.draw.circle(field, rng.choice(FLOWER_COLORS), pos, 1.5)

    lo, hi_x, hi_y = FENCE_INSET, SCREEN_RES_X - FENCE_INSET, SCREEN_RES_Y - FENCE_INSET
    corners = [Vector2(lo, lo), Vector2(hi_x, lo), Vector2(hi_x, hi_y), Vector2(lo, hi_y)]
    for a, b in zip(corners, corners[1:] + corners[:1]):
        draw_fence(field, a, b)
    for a, b in pen_walls(gate_open=True):  # The gate itself is drawn every frame.
        draw_fence(field, a, b)
    return field


def make_herd() -> list[Sheep]:
    """A few loose groups scattered over the pasture, clear of the pen, and
    NUM_LAMBS lambs each next to its mother."""

    def clear(p: Vector2) -> bool:
        return (_FIELD_MIN + 20 < p.x < _FIELD_MAX_X - 20 and _FIELD_MIN + 20 < p.y < _FIELD_MAX_Y - 20
                and not (p.x > _PEN_LEFT - 30 and p.y > _PEN_TOP - 30))

    sheep = []
    groups = 6
    for g in range(groups):
        centre = Vector2()
        while not clear(centre):
            centre = Vector2(random.uniform(300, SCREEN_RES_X - 300),
                             random.uniform(250, SCREEN_RES_Y - 250))
        size = NUM_SHEEP // groups + (1 if g < NUM_SHEEP % groups else 0)
        for _ in range(size):
            pos = Vector2()
            while not clear(pos):
                pos = centre + Vector2(random.gauss(0, 60), random.gauss(0, 60))
            sheep.append(Sheep(pos))
    for mother in random.sample(sheep, NUM_LAMBS):
        pos = Vector2()
        while not clear(pos):
            pos = mother.pos + Vector2(random.uniform(15, 30), 0).rotate(random.uniform(0, 360))
        sheep.append(Sheep(pos, mother))
    return sheep


def run() -> None:
    """Main loop."""
    pygame.init()
    pygame.display.set_caption("Sheep")
    screen = pygame.display.set_mode((SCREEN_RES_X, SCREEN_RES_Y))
    clock = pygame.time.Clock()
    pasture = make_pasture()
    night = pygame.Surface((SCREEN_RES_X, SCREEN_RES_Y), pygame.SRCALPHA)
    font = pygame.font.Font(None, 26)

    world = World()
    show_hunger = False

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_h:
                show_hunger = not show_hunger
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_g:
                world.toggle_gate()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                world = World()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                world.click_dog(Vector2(event.pos))
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                world.click_hay(Vector2(event.pos))

        world.step()
        world.draw(screen, pasture, night, font, show_hunger)
        pygame.display.flip()

        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    run()
