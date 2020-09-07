import pygame
from settings import *

class RayCast:
    def __init__(self, world_map):
        self.world_map = world_map
        self.ox = 0
        self.oy = 0
        self.endx = [0 for _ in range(NUM_RAYS)]
        self.endy = [0 for _ in range(NUM_RAYS)]
        
        self.depth = [0 for _ in range(NUM_RAYS)]

    def draw(self, sc):
        if self.endx[-1] != 0.0:
            for idx, x in enumerate(self.endx):
                pygame.draw.line(sc, RAYS_COLOR_1, (self.ox, self.oy), (x, self.endy[idx]), 2)

    def mapping(self, a, b):
        return (a // TILE) * TILE, (b // TILE) * TILE

    def ray_casting(self, player_pos, player_angle):
        self.ox, self.oy = player_pos
        xm, ym = self.mapping(self.ox, self.oy)
        cur_angle = player_angle - HALF_FOV
        for ray in range(NUM_RAYS):
            sin_a = math.sin(cur_angle)
            cos_a = math.cos(cur_angle)
            sin_a = sin_a if sin_a else 0.000001
            cos_a = cos_a if cos_a else 0.000001

            # verticals
            x, dx = (xm + TILE, 1) if cos_a >= 0 else (xm, -1)
            for _ in range(0, WIDTH, TILE):
                depth_v = (x - self.ox) / cos_a
                y = self.oy + depth_v * sin_a
                tile_v = self.mapping(x + dx, y)
                if depth_v > MAX_DEPTH:
                    break
                if tile_v in self.world_map:
                    if self.world_map[tile_v] == '1':
                        break
                x += dx * TILE

            # horizontals
            y, dy = (ym + TILE, 1) if sin_a >= 0 else (ym, -1)
            for _ in range(0, HEIGHT, TILE):
                depth_h = (y - self.oy) / sin_a
                x = self.ox + depth_h * cos_a
                tile_h = self.mapping(x, y + dy)
                if depth_h > MAX_DEPTH:
                    break
                if tile_h in self.world_map:
                    if self.world_map[tile_h] == '1':
                        break
                y += dy * TILE
            
            self.depth[ray] = max(depth_v if depth_v < depth_h else depth_h, 0.00001)
            
            self.endx[ray] = self.ox + self.depth[ray] * cos_a
            self.endy[ray] = self.oy + self.depth[ray] * sin_a

            cur_angle += DELTA_ANGLE
