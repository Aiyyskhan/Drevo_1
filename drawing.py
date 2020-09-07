import pygame
from settings import *
# from map_file import get_map

class Drawing:
    def __init__(self, sc, sc_map):
        self.sc = sc
        self.sc_map = sc_map
        self.font = pygame.font.SysFont('Arial', 20, bold=False)

    def background(self):
        pygame.draw.rect(self.sc, BACK_COLOR_1, (0, 0, WIDTH, HEIGHT))
        # pygame.draw.rect(self.sc, DARKGRAY, (0, HALF_HEIGHT, WIDTH, HALF_HEIGHT))

    def info(self, num_generation, num_winners, num_live_players, clock):
        render = self.font.render(f"Generation: {int(num_generation)}", 0, TEXT_COLOR_1)
        self.sc.blit(render, GENERATION_INFO_POS)
        render = self.font.render(f"Winners: {int(num_winners)}", 0, TEXT_COLOR_1)
        self.sc.blit(render, NUM_WINNERS_INFO_POS)
        render = self.font.render(f"Players: {int(num_live_players)}", 0, TEXT_COLOR_1)
        self.sc.blit(render, NUM_PLAYERS_INFO_POS)
        render = self.font.render(f"FPS: {int(clock.get_fps())}", 0, TEXT_COLOR_1)
        self.sc.blit(render, FPS_INFO_POS)

    # def mini_map(self, player):
    #     self.sc_map.fill(BLACK)
    #     map_x, map_y = player.x // MAP_SCALE, player.y // MAP_SCALE
    #     pygame.draw.line(self.sc_map, YELLOW, (map_x, map_y), (map_x + 12 * math.cos(player.angle),
    #                                              map_y + 12 * math.sin(player.angle)), 2)
    #     pygame.draw.circle(self.sc_map, RED, (int(map_x), int(map_y)), 5)
    #     for x, y in mini_map:
    #         pygame.draw.rect(self.sc_map, GREEN, (x, y, MAP_TILE, MAP_TILE))
    #     self.sc.blit(self.sc_map, MAP_POS)