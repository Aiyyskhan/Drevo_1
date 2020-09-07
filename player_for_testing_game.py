from settings import *
import pygame
import math
import numpy as np

class Player:
    def __init__(self, sc, collision_walls, finish_set):
        self.sc = sc
        self.max_speed = PLAYER_SPEED
        self.max_angular_velocity = PLAYER_ANGULAR_VELOCITY
        
        self.color = GREEN
        self.radius = 12
        self.init_angle = 0

        # collision parameters
        self.side = self.radius * 2
        self.collision_list = collision_walls

        self.rays = None
        self.brain = None

        self.finish_coords = finish_set

    def setup(self):
        self.x = TILE + TILE // 2
        self.y = HEIGHT - (TILE + TILE // 2)
        self.init_x = self.x
        self.init_y = self.y
        self.prev_x = self.x
        self.prev_y = self.y
        self.dx = 0
        self.dy = 0
        self.angle = self.init_angle
        self.rect = pygame.Rect(self.x, self.y, self.side, self.side)

    @property
    def pos(self):
        return (self.x, self.y)        

    def detect_collision(self):
        next_rect = self.rect.copy()
        next_rect.move_ip(self.dx, self.dy)
        hit_indexes = next_rect.collidelistall(self.collision_list)
        
        if len(hit_indexes):
            self.collision_flag = True
            delta_x, delta_y = 0, 0
            for hit_index in hit_indexes:
                hit_rect = self.collision_list[hit_index]
                if self.dx > 0:
                    delta_x += next_rect.right - hit_rect.left
                else:
                    delta_x += hit_rect.right - next_rect.left
                if self.dy > 0:
                    delta_y += next_rect.bottom - hit_rect.top
                else:
                    delta_y += hit_rect.bottom - next_rect.top

            if abs(delta_x - delta_y) < 10:
                self.dx, self.dy = 0, 0
            elif delta_x > delta_y:
                self.dy = 0
            elif delta_y > delta_x:
                self.dx = 0        

    def movement(self):
        if self.brain != None:
            self.rays.ray_casting((self.x, self.y), self.angle)
            self.behavior_control()
            self.detect_collision()
            self.x += self.dx
            self.y += self.dy
            self.rect.center = self.x, self.y

    def behavior_control(self):
        act_thrust, act_left, act_right = self.brain( np.array(self.rays.depth) / MAX_DEPTH )

        self.angle += (self.max_angular_velocity * act_left) - (self.max_angular_velocity * act_right)

        sin_a = math.sin(self.angle)
        cos_a = math.cos(self.angle)

        self.dx = self.max_speed * act_thrust * cos_a
        self.dy = self.max_speed * act_thrust * sin_a

    def draw(self):
        self.rays.draw(self.sc)
        pygame.draw.circle(self.sc, self.color, (int(self.x), int(self.y)), self.radius)
        pygame.draw.line(self.sc, self.color, (self.x, self.y), (self.x + 30 * math.cos(self.angle), self.y + 30 * math.sin(self.angle)), 2)