
import ganglion_20200831 as g
import numpy as np


import pygame
from settings import *
from player_for_testing_game import Player
import math
from map_file_lev0_2 import get_map
from drawing import Drawing
from ray_casting import RayCast

class Game:
    def __init__(self):
        self.color_r = np.random.randint(50, 220) # np.linspace(10, 200).astype(int)
        self.color_g = np.random.randint(50, 220) # np.linspace(10, 200).astype(int)
        self.color_b = np.random.randint(50, 220) # np.linspace(200, 10).astype(int)


        pygame.init()
        self.sc = pygame.display.set_mode((WIDTH, HEIGHT))
        self.sc_map = pygame.Surface((WIDTH // MAP_SCALE, HEIGHT // MAP_SCALE))
        self.clock = pygame.time.Clock()
        self.drawing = Drawing(self.sc, self.sc_map)

        self.road_coords = set()
        self.finish_coords = set()
        self.wall_coord_list = list()
        self.world_map, self.collision_walls = get_map(TILE)
        for coord, signature in self.world_map.items():
            if signature == "1":
                self.wall_coord_list.append(coord)
            elif signature == "2":
                self.finish_coords.add(coord)
            elif signature == ".":
                self.road_coords.add(coord)

        self.main_save_path = "data/"
        self.load_path = "data/level_5_and_0_20200901/individ_5.3.30.gen1_2.npy" #"data/level_3_20200901/individ_5.3.30.gen7_7.npy" # "data/level_2_20200901/individ_5.3.30.gen5_24.npy" #"data/individ_6_5330_gen8.npy"
        self.save_number = 4
        self.load_number = 0

    def player_setup(self):
        self.loading()

        self.player = Player(self.sc, self.collision_walls, self.finish_coords)
        self.player.color = (self.color_r, self.color_g, self.color_b)
        self.player.init_angle = math.pi + (math.pi/2)
        self.player.rays = RayCast(self.world_map)
        
        self.player.brain = g.Ganglion(self.loaded_parameters[0], self.loaded_parameters[1], self.loaded_parameters[2], self.loaded_parameters[3])
        self.player.brain.index_list = self.loaded_indices
        self.player.brain.synapse_weight_list = self.loaded_weights
        self.player.test_mode = True

        self.player.setup()


    def game_event(self):
        # drawing.background()

        self.player.movement()
        self.player.draw()
        
        for x, y in self.wall_coord_list:
            pygame.draw.rect(self.sc, WALL_COLOR_1, (x, y, TILE, TILE), 2)
        for x, y in self.finish_coords:
            pygame.draw.rect(self.sc, WALL_COLOR_2, (x, y, TILE, TILE), 2)

        self.drawing.info(0, 0, 1, self.clock)

    def loading(self):
        with open(self.load_path, 'rb') as f:
            len_weights = np.load(f, allow_pickle=True)
            self.loaded_parameters = np.load(f, allow_pickle=True)
            self.loaded_indices = np.load(f, allow_pickle=True)
            self.loaded_weights = []
            for _ in range(len_weights):
                self.loaded_weights.append(np.load(f, allow_pickle=True))
        
        self.load_number += 1

    def run(self):
        self.player_setup()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
            self.sc.fill(BLACK)

            self.game_event()

            pygame.display.flip()
            self.clock.tick()


if __name__ == "__main__":
    game = Game()
    game.run()