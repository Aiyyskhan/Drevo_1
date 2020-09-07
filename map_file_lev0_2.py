import pygame

text_map = [
    '111111111111111',
    '1.....111....21',
    '1.111..11.11121',
    '1....1..1...111',
    '1111.11.111...1',
    '1....1...1.11.1',
    '1.11.111......1',
    '111111111111111'
]

def get_map(tile_size):
    world_map = {}
    collision_walls = []
    for j, row in enumerate(text_map):
        for i, char in enumerate(row):
            if char != '.':
                if char == '1':
                    world_map[(i * tile_size, j * tile_size)] = '1'
                    collision_walls.append(pygame.Rect(i * tile_size, j * tile_size, tile_size, tile_size))
                elif char == '2':
                    world_map[(i * tile_size, j * tile_size)] = '2'
    return world_map, collision_walls