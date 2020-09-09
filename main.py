import genetic_algorithm_20200831 as ga
import ganglion_20200831 as g
import numpy as np


import pygame
from settings import *
from player_base import Player
import math
from map_file_lev6 import get_map
from drawing import Drawing
from ray_casting import RayCast

ORIGIN_MODE = False # False если эволюция инициируется от сохраненных родителей
NEURON_ADDITION_MODE = False
SAVING_WINNER = False

class Game:
    def __init__(self):
        self.color_r = np.random.randint(50, 220, NUM_PLAYERS) #np.linspace(126, 194, NUM_PLAYERS).astype(int)
        self.color_g = np.random.randint(50, 220, NUM_PLAYERS) #np.linspace(130, 114, NUM_PLAYERS).astype(int)
        self.color_b = np.random.randint(50, 220, NUM_PLAYERS) #np.linspace(45, 29, NUM_PLAYERS).astype(int)

        self.game_over = False
        self.generation = 1
        self.epoch = 0
        self.max_epoch = 10000
        
        self.number_of_live_players = NUM_PLAYERS
        self.number_of_winners = 0

        self.mutation_time = False

        self.max_num_stagnation = 10
        self.num_stagnation = 0

        self.best_players = []
        self.best_player_reward = 0.0

        self.main_load_path = "data/20_hid_neurons/level_6_20200903/individ_5.3.20"
        self.main_save_path = "data/20_hid_neurons/level_6_20200903/"

        self.save_number = 0
        self.load_number = 0

        pygame.init()
        self.sc = pygame.display.set_mode((WIDTH, HEIGHT))
        self.sc_map = pygame.Surface((WIDTH // MAP_SCALE, HEIGHT // MAP_SCALE))
        self.clock = pygame.time.Clock()
        self.drawing = Drawing(self.sc, self.sc_map)

        self.finish_coords = set()
        self.wall_coord_list = list()
        self.world_map, self.collision_walls = get_map(TILE)
        for coord, signature in self.world_map.items():
            if signature == "1":
                self.wall_coord_list.append(coord)
            elif signature == "2":
                self.finish_coords.add(coord)
        
        self.angle_diff = 360 // NUM_PLAYERS
        self.players = []
        
        for i in range(NUM_PLAYERS):
            player = Player(self.sc, self.collision_walls, self.finish_coords)
            player.color = (self.color_r[i], self.color_g[i], self.color_b[i])
            player.init_angle = math.pi + (math.pi/2) #self.angle_diff * i
            player.rays = RayCast(self.world_map)
            player.brain = g.Ganglion(NUM_RAYS, 3, 20, 500)
            player.setup()
            self.players.append(player)

        # for level 1
        # self.load_paths = [
        #     "data/level_0_20200901/individ_5.3.30.gen3_1.npy",
        #     "data/level_0_20200901/individ_5.3.30.gen6_4.npy",
        #     "data/level_0_20200901/individ_5.3.30.gen7_5.npy",
        #     "data/level_0_20200901/individ_5.3.30.gen8_6.npy",
        #     "data/level_0_20200901/individ_5.3.30.gen11_9.npy",
        #     "data/level_0_20200901/individ_5.3.30.gen12_10.npy",
        #     "data/level_0_20200901/individ_5.3.30.gen14_12.npy",
        #     "data/level_0_20200901/individ_5.3.30.gen16_13.npy",
        #     "data/level_0_20200901/individ_5.3.30.gen17_14.npy",
        #     "data/level_0_20200901/individ_5.3.30.gen18_15.npy",
        #     "data/level_0_20200901/individ_5.3.30.gen19_16.npy",
        #     "data/level_0_20200901/individ_5.3.30.gen20_17.npy"
        # ] #"data/individ_6_5330_gen8.npy"

        # for level 2 & 3
        # self.load_paths = [
        #     "data/level_1_20200901/individ_5.3.30.gen11_4.npy",
        #     "data/level_1_20200901/individ_5.3.30.gen13_6.npy",
        #     "data/level_1_20200901/individ_5.3.30.gen14_7.npy",
        #     "data/level_1_20200901/individ_5.3.30.gen15_9.npy",
        #     "data/level_1_20200901/individ_5.3.30.gen19_15.npy",
        #     "data/level_1_20200901/individ_5.3.30.gen20_17.npy",
        #     "data/level_1_20200901/individ_5.3.30.gen21_20.npy",
        #     "data/level_1_20200901/individ_5.3.30.gen24_25.npy",
        #     "data/level_1_20200901/individ_5.3.30.gen27_30.npy",
        #     "data/level_1_20200901/individ_5.3.30.gen29_35.npy",
        #     "data/level_1_20200901/individ_5.3.30.gen33_40.npy",
        #     "data/level_1_20200901/individ_5.3.30.gen34_45.npy",
        #     "data/level_1_20200901/individ_5.3.30.gen37_50.npy",
        #     "data/level_1_20200901/individ_5.3.30.gen37_53.npy",
        #     "data/level_1_20200901/individ_5.3.30.gen38_54.npy",
        #     "data/level_1_20200901/individ_5.3.30.gen39_55.npy"
        # ]

        # # for level 4
        # self.load_paths = [
        #     "data/level_1_20200901/individ_5.3.30.gen11_4.npy",
        #     "data/level_1_20200901/individ_5.3.30.gen13_6.npy",
        #     "data/level_1_20200901/individ_5.3.30.gen14_7.npy",
        #     "data/level_1_20200901/individ_5.3.30.gen15_9.npy",
        #     "data/level_1_20200901/individ_5.3.30.gen19_15.npy",
        #     "data/level_1_20200901/individ_5.3.30.gen20_17.npy",
        #     "data/level_1_20200901/individ_5.3.30.gen21_20.npy",
        #     "data/level_1_20200901/individ_5.3.30.gen24_25.npy",
        #     "data/level_1_20200901/individ_5.3.30.gen34_45.npy",
        #     "data/level_1_20200901/individ_5.3.30.gen37_50.npy",
        #     "data/level_1_20200901/individ_5.3.30.gen37_53.npy",
        #     "data/level_1_20200901/individ_5.3.30.gen38_54.npy",
        #     "data/level_1_20200901/individ_5.3.30.gen39_55.npy",
        #     "data/level_2_20200901/individ_5.3.30.gen2_15.npy",
        #     "data/level_2_20200901/individ_5.3.30.gen4_22.npy",
        #     "data/level_2_20200901/individ_5.3.30.gen4_23.npy",
        #     "data/level_2_20200901/individ_5.3.30.gen5_24.npy",
        #     "data/level_3_20200901/individ_5.3.30.gen3_5.npy",
        #     "data/level_3_20200901/individ_5.3.30.gen5_6.npy",
        #     "data/level_3_20200901/individ_5.3.30.gen7_7.npy"
        # ]

        # for level 0_2 and player lev4_1
        # self.load_paths = [
        #     "data/level_4_20200901/individ_5.3.30.gen6_4.npy",
        #     "data/level_4_20200901/individ_5.3.30.gen7_5.npy",
        #     "data/level_4_20200901/individ_5.3.30.gen7_6.npy",
        #     "data/level_4_20200901/individ_5.3.30.gen7_7.npy",
        #     "data/level_4_20200901/individ_5.3.30.gen7_8.npy",
        #     "data/level_4_20200901/individ_5.3.30.gen8_9.npy",
        #     "data/level_4_20200901/individ_5.3.30.gen8_10.npy"
        # ]

        self.load_paths = []
        for i in range(0, 50):
            path = f"{self.main_load_path}_{i}.npy"
            self.load_paths.append(path)

        # self.main_load_path = "data/20_hid_neurons/level_1_1_20200903/individ_5.3.20"
        # for i in range(59, 70):
        #     path = f"{self.main_load_path}_{i}.npy"
        #     self.load_paths.append(path)

        if not ORIGIN_MODE:
            num_leaders = len(self.load_paths)
            self.number_of_live_players = num_leaders

            for idx, path in enumerate(self.load_paths):
                parameters, indices, weights = self.loading(path)
                self.players[idx].brain = g.Ganglion(parameters[0], parameters[1], parameters[2], parameters[3])
                self.players[idx].brain.index_list = indices
                self.players[idx].brain.synapse_weight_list = weights
                
                # self.players[idx].brain.add_units()
            for player in self.players[num_leaders:]:
                player.dead = True
                player.verified = True

    def game_event(self):
        # drawing.background()
        for player in self.players:
            if player.dead and not player.verified:
                self.number_of_live_players -= 1
                player.verified = True
            else:
                player.movement()
                player.draw()

            if player.reached_finish=="y":
                player.reached_finish = "v"
                self.number_of_winners += 1
                self.best_player_reward = 0.0
                self.num_stagnation = 0
        
        for x, y in self.wall_coord_list:
            pygame.draw.rect(self.sc, WALL_COLOR_1, (x, y, TILE, TILE), 2)
        for x, y in self.finish_coords:
            pygame.draw.rect(self.sc, WALL_COLOR_2, (x, y, TILE, TILE), 2)

        self.drawing.info(self.generation, self.number_of_winners, self.number_of_live_players, self.clock)
        # drawing.mini_map(player)

    def stop_function(self):
        if self.number_of_live_players == 0 or self.epoch >= self.max_epoch:
            self.game_over = True
            self.epoch = 0
        if self.game_over and self.number_of_live_players > 0:
            for player in self.players:
                if not player.dead:
                    self.saving(player)
                    self.number_of_winners += 1
                    self.best_player_reward = 0.0
                    self.num_stagnation = 0


    def saving(self, player):
        parameters, indices = player.brain.get_parameters()
        weights = player.brain.get_synapse_weights()
        len_weights = len(weights)

        path = f"{self.main_save_path}individ_{parameters[0]}.{parameters[1]}.{parameters[2]}_{self.save_number}.npy"

        with open(path, 'wb') as f:
            np.save(f, len_weights)
            np.save(f, parameters)
            np.save(f, indices)
            for w_array in weights:
                np.save(f, w_array)

        self.save_number += 1

    def loading(self, path):
        with open(path, 'rb') as f:
            len_weights = np.load(f, allow_pickle=True)
            
            parameters = np.load(f, allow_pickle=True)
            indices = np.load(f, allow_pickle=True)
            weights = []
            
            for _ in range(len_weights):
                weights.append(np.load(f, allow_pickle=True))
        return parameters, indices, weights
            

    def evolution(self):
        result_list = []
        parent_weight_list = []
        leader_weights = None
        child_weights = None

        upgrade_flag = False
        if NEURON_ADDITION_MODE and self.num_stagnation == self.max_num_stagnation:
            upgrade_flag = True
            self.best_player_reward = 0.0
            self.num_stagnation = 0    

        for idx, player in enumerate(self.players):
            result_list.append(player.reward)

            if SAVING_WINNER and player.reached_finish == "v":
                self.saving(player)
        
        if upgrade_flag:
            print("Upgrade!", self.players[0].brain.num_units+1)

        if max(result_list) >= self.best_player_reward:
            self.best_players = self.players.copy()
            self.best_player_reward = max(result_list)
            self.generation += 1

            # if self.generation % 50 == 0:
            #     upgrade_flag = True
        else:
            self.players = self.best_players.copy()
            result_list.clear()
            for player in self.players:
                result_list.append(player.reward)

            if NEURON_ADDITION_MODE:
                if self.num_stagnation < self.max_num_stagnation:
                    self.num_stagnation += 1        

        for idx, player in enumerate(self.players):
            parent_weight_list.append(player.brain.get_synapse_weights())

        if not self.mutation_time:
            leader_weights = ga.selection(parent_weight_list, result_list.copy(), NUM_WINNERS, True)
                
            child_weights = ga.crossover(leader_weights, NUM_PLAYERS)
            
            self.mutation_time = True
        else:
            child_weights = parent_weight_list.copy()
            child_weights = ga.mutation(child_weights)
            
            self.mutation_time = False

        for idx, player in enumerate(self.players):
            player.brain.synapse_weight_list = child_weights[idx].copy()
            
            if upgrade_flag:
                player.brain.add_units()
            
            player.setup()
                
        self.number_of_live_players = NUM_PLAYERS
        self.game_over = False


    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
            self.sc.fill(BLACK)

            if not self.game_over:
                self.game_event()
                self.stop_function()
            else:
                self.evolution()

            pygame.display.flip()
            self.clock.tick()
            self.epoch += 1


if __name__ == "__main__":
    game = Game()
    game.run()