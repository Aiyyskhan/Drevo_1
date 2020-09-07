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

        self.test_mode = False

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

        self.dead = False
        self.verified = False
        self.reached_finish = "n"

        self.distance_traveled = 0.0
        self.path_radius = 0.0
        self.radius_treshold = 140.0 #50.0
        self.reward = 0.0
        self.number_reversal = 0
        
        self.counter_1 = 0
        self.radius_check_interval = 500 #10

        self.num_collision = 15
        self.collision_flag = False
        self.counter_2 = 0
        self.collision_check_interval = 100

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

        # if len(hit_indexes): 
        #     self.dead = True

    def movement(self):
        if not self.dead and self.brain != None:
            self.rays.ray_casting((self.x, self.y), self.angle)
            self.behavior_control()
            self.detect_collision()
            self.x += self.dx
            self.y += self.dy
            self.rect.center = self.x, self.y
            
            if not self.test_mode:
                # dist = math.sqrt((self.x-self.init_x)**2 + (self.y-self.init_y)**2)
                # if dist > self.reward:
                #     self.reward = dist
                self.reward += 0.1
                x, y = (self.x // TILE) * TILE, (self.y // TILE) * TILE
                if (x, y) in self.finish_coords and self.reached_finish=="n":
                    self.reached_finish = "y"

                if self.counter_1 >= self.radius_check_interval:
                    self.distance_calculation()
                    self.path_calculation()

                    self.counter_1 = 0
                self.counter_1 += 1

                # if self.counter_2 >= self.collision_check_interval:
                #     if self.collision_flag:
                #         self.reward -= 1
                #         if self.num_collision:
                #             self.num_collision -= 1
                #         else:
                #             self.dead = True
                #         self.collision_flag = False
                #     self.counter_2 = 0
                # self.counter_2 += 1


    def behavior_control(self):
        act_thrust, act_left, act_right = self.brain( np.array(self.rays.depth) / MAX_DEPTH )

        self.angle += (self.max_angular_velocity * act_left) - (self.max_angular_velocity * act_right)

        sin_a = math.sin(self.angle)
        cos_a = math.cos(self.angle)

        self.dx = self.max_speed * act_thrust * cos_a
        self.dy = self.max_speed * act_thrust * sin_a

    def draw(self):
        if not self.dead:
            self.rays.draw(self.sc)
            pygame.draw.circle(self.sc, self.color, (int(self.x), int(self.y)), self.radius)
            pygame.draw.line(self.sc, self.color, (self.x, self.y), (self.x + 30 * math.cos(self.angle), self.y + 30 * math.sin(self.angle)), 2)

    def path_calculation(self):
        if self.distance_traveled <= self.radius_treshold:
            self.dead = True
        else:
            # self.reward += self.distance_traveled * 0.2
            self.distance_traveled = 0.0
    
    def distance_calculation(self):
        self.distance_traveled += math.sqrt((self.x-self.prev_x)**2 + (self.y-self.prev_y)**2)
        self.prev_x = self.x
        self.prev_y = self.y