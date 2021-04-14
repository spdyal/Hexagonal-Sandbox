import pygame
import math
import random
import playsound
import os
import sys
from utilities import *

# Объявление констант
SQRT_3 = math.sqrt(3)
CHUNK_SIZE = 16
SCALE = 16
MAX_LIGHT = 8
SPEED = 0.2
GRAVITY = 1

# Кэш текстур запоминает сгенерированные текстуры чанков и тайлов
TEXTURE_CACHE = {}

# Класс генератора текстур тайлов по шаблону
class HexagonTextureRenderer:
    def __init__(self, size, texture=None, variated=True):
        self.triangles = 6 * (size ** 2)
        self.size = size
        self.variated = variated
        if type(texture) == list:
            self.texture = texture
        elif type(texture) == tuple:
            self.texture = [texture for _ in range(self.triangles)]
        elif texture is None:
            self.texture = [(0, 0, 0) for _ in range(self.triangles)]
        else:
            raise TypeError

    def render(self, scale, quality=1, smooth=False):

        triangle_length = scale * quality
        output = pygame.Surface((4 * triangle_length,
                                 5 * triangle_length),
                        pygame.SRCALPHA, 32)
        output.convert_alpha()

        h = (triangle_length * SQRT_3) / 2
        gx, gy = (self.size * triangle_length, h * self.size)
        tpl = self.size
        fall = False
        index = 0
        for i in range(self.size * 2):
            dy = (self.size - i) * h
            dx = triangle_length * (tpl / 2)
            for j in range(tpl):
                index += 1
                color1 = self.texture[index - 1]
                color2 = self.texture[index - 1 + self.triangles // 2]
                b = triangle_length * j
                if not color1 is None:
                    if self.variated:
                        color1 = variate_color(color1, (-5, 7))
                    pygame.draw.polygon(output, color1,
                                        [(gx - dx + b, gy - dy),
                                         (gx - dx + b + triangle_length / 2, gy - dy + h),
                                         (gx - dx + b + triangle_length, gy - dy)])

                if not color2 is None:
                    if self.variated:
                        color2 = variate_color(color2, (-5, 7))
                        pass
                    pygame.draw.polygon(output, color2,
                                        [(gx - dx + b, gy + dy),
                                         (gx - dx + b + triangle_length / 2, gy + dy - h),
                                         (gx - dx + b + triangle_length, gy + dy)])
            fall = fall or tpl >= self.size * 2
            if fall:
                tpl -= 1
            else:
                tpl += 1
        if smooth:
            return pygame.transform.smoothscale(output, (4 * scale,
                                                         5 * scale))
        else:
            return pygame.transform.scale(output, (4 * scale,
                                                   5 * scale))


# Класс частицы
class Particle(pygame.sprite.Sprite):
    def __init__(self, camera, texture, world_coords, dx, dy, weight=1):
        super().__init__(camera.particle_layer)
        self.image = texture
        self.rect = self.image.get_rect()
        self.velocity = Vector2(dx, dy)
        self.weight = weight
        self.world_coords = world_coords
        self.rect.x, self.rect.y = camera.apply_coords(world_coords)
        self.camera = camera

    def update(self):
        self.velocity.y += GRAVITY * self.weight / 100
        self.world_coords += self.velocity * self.weight
        self.rect.x, self.rect.y = self.camera.apply_coords(self.world_coords)
        if not self.rect.colliderect(SCREEN_BORDER):
            self.kill()

# Функция воспроизведения звука из ассетов
def sound(name):
    if name == 'None':
        return
    try:
        playsound.playsound('assets/sounds/' + name + '.mp3', False)
    except Exception:
        pass

# Инициализация игрового контента
# | Материалы
MATERIALS = [None]
class MaterialData:
    name = 'Abstract'
    color = (255, 0, 0)
    sound = 'cobble'
    illumination = 0
    fertile = True
    
    def __init__(self):
        self.id = len(MATERIALS)
        MATERIALS.append(self)
    
AIR = MaterialData()
AIR.name = 'Air'
AIR.color = None
AIR.sound = 'None'
AIR.illumination = MAX_LIGHT + 1

GRASS = MaterialData()
GRASS.name = 'Grass'
GRASS.color = (100, 255, 100)
GRASS.sound = 'moss'

LICHEN = MaterialData()
LICHEN.name = 'Lichen'
LICHEN.color = (103, 163, 141)
LICHEN.sound = 'moss'

DIRT = MaterialData()
DIRT.name = 'Dirt'
DIRT.color = (153, 82, 0)
DIRT.sound = 'soil'

STONE = MaterialData()
STONE.name = 'Stone'
STONE.color = (143, 161, 159)
STONE.fertile = False

SNOW = MaterialData()
SNOW.name = 'Snow'
SNOW.color = (199, 237, 252)
SNOW.sound = 'soil'

COLD_DIRT = MaterialData()
COLD_DIRT.name = 'Cold Dirt'
COLD_DIRT.color = (117, 77, 40)
COLD_DIRT.sound = 'soil'

SAND = MaterialData()
SAND.name = 'Sand'
SAND.color = (255, 234, 158)
SAND.sound = 'moss'

GRANITE = MaterialData()
GRANITE.name = 'Granite'
GRANITE.color = (207, 149, 149)
GRANITE.fertile = False

HUMUS = MaterialData()
HUMUS.name = 'Humus'
HUMUS.color = (204, 146, 29)
HUMUS.sound = 'soil'

GLOW_GRASS = MaterialData()
GLOW_GRASS.name = 'GLOW_GRASS'
GLOW_GRASS.color = (0, 75, 224)
GLOW_GRASS.sound = 'wet'
GLOW_GRASS.illumination = 5

# | Биомы
BIOMES = []
class BiomeData:
    climates = range(60)
    soil = DIRT
    flora = GRASS
    skycolor = (184, 255, 250)

    def __init__(self):
        BIOMES.append(self)

PLAINS = BiomeData()
PLAINS.climates = range(20, 100)

TUNDRA = BiomeData()
TUNDRA.climates = range(0, 10)
TUNDRA.soil = COLD_DIRT
TUNDRA.flora = SNOW
TUNDRA.skycolor = (96, 134, 150)

DESERT = BiomeData()
DESERT.climates = range(200, 255)
DESERT.soil = SAND
DESERT.flora = None
DESERT.skycolor = (223, 245, 164)

MOUNTAINS = BiomeData()
MOUNTAINS.climates = list(range(10, 20)) + list(range(100, 150))
MOUNTAINS.soil = STONE
MOUNTAINS.flora = LICHEN

GRANITE_MOUNTAINS = BiomeData()
GRANITE_MOUNTAINS.climates = range(150, 200)
GRANITE_MOUNTAINS.soil = GRANITE
MOUNTAINS.flora = LICHEN

GLOW_LANDS = BiomeData()
GLOW_LANDS.climates = [255]
GLOW_LANDS.soil = HUMUS
GLOW_LANDS.flora = GLOW_GRASS
GLOW_LANDS.skycolor = (26, 51, 102)

# | Формы тайлов
SHAPES = []
class ShapeData:
    texture_template = [1] * 24
    is_empty = False
    
    def __init__(self):
        self.id = len(SHAPES)
        SHAPES.append(self)

VOID = ShapeData()
VOID.texture_template = [None] * 24
VOID.is_empty = True

SOLID = ShapeData()

SLAB = ShapeData()
SLAB.texture_template = [2] * 5 + [1] * 12 + [2] * 7

REVERSE_SLAB = ShapeData()
REVERSE_SLAB.texture_template = [1] * 5 + [2] * 12 + [1] * 7

WALL = ShapeData()
t1, t2 = [(1, -40, -40, -40)], [(1, -25, -25, -25)]
WALL.texture_template = t1 * 3 + t2 + t1 * 2 + t2 * 2 + t1 * 8 + t2 * 2 + t1 * 2 + t2 + t1 * 3
WALL.is_empty = True
del t1, t2

MOSSED = ShapeData()
MOSSED.texture_template = [2] * 10 + [1] + [2] + [1] * 2 + [2] + [1] + [2] * 8

# Класс тайла
class Tile:
    # | Инициализация принимает родительский чанк, базовый материал, форму и, если надо, покрывающий материал
    def __init__(self, chunk, material, shape, cover=None):
        self.material = material
        self.shape = shape
        self.chunk = chunk
        self.cover = cover
        self.light = 0
        
        if shape.is_empty:
            self.light_res = 0.5
        else:
            self.light_res = 1

    # | Рендер текстуры тайла в зависимости от цветов базового и покрывающего материала, шаблона текстуры формы,
    # | а так же уровня освещения тайла. В случае, если такая текстура уже была сгенерированна, возвращает
    # | найденную подходящую текстуру в кэше текстур.
    def render(self, scale):
        illumination = self.get_illumination()
        key = str(hash(self)) + str(scale)
        shade_key = str(hash((0, 0, 0))) + str(scale)

        if shade_key not in TEXTURE_CACHE:
            TEXTURE_CACHE[shade_key] = HexagonTextureRenderer(2, (0, 0, 0), False).render(scale, 2)

        if illumination == 0:
            return TEXTURE_CACHE[shade_key]

        if key not in TEXTURE_CACHE:
            texture = []
            for color in self.shape.texture_template:
                if color is None:
                    texture.append(None)
                    continue
                if color == 1:
                    texture.append(self.material.color)
                elif color == 2 and self.cover:
                    texture.append(self.cover.color)
                elif color == 2:
                    texture.append(None)
                elif isinstance(color, tuple):
                    text, r, g, b = color
                    if text == 1:
                        mr, mg, mb = self.material.color
                        texture.append(stabilize_color(mr + r,
                                                       mg + g,
                                                       mb + b))
                else:
                    texture.append(color)
            TEXTURE_CACHE[key] = HexagonTextureRenderer(2, texture).render(scale, 2)
        if illumination >= MAX_LIGHT:
            return TEXTURE_CACHE[key]

        output = TEXTURE_CACHE[shade_key].copy()
        over = TEXTURE_CACHE[key].copy()
        alpha = (illumination / MAX_LIGHT) * 255
        over.set_alpha(alpha)
        output.blit(over, (0, 0))

        return output

    # | Хэш тайла такого типа, используется как ключ для хранения текстуры в кэше текстур
    def __hash__(self):
        if self.cover:
            return hash('C' + str(id(self.material)) + str(id(self.cover.name)) + str(id(self.shape)))
        return hash(str(id(self.material)) + str(id(self.shape)))

    # | Проверка на необходимость хранения некоторых параметров тайла (например, не является ли
    # | тайл полублоком воздуха)
    def void_test(self):
        if self.shape == VOID:
            self.material = AIR
            self.cover = None
        if self.material == AIR and (not self.cover or self.cover == AIR):
            self.shape = VOID
        if self.cover == self.material:
            self.shape = SOLID
            self.cover = None

    # | Быстрая смена формы тайла
    def reshape(self, new_shape):
        self.shape = new_shape
        self.light_res = 0.5 if new_shape.is_empty else 1
        self.void_test()

    # | Быстрая смена базового материала тайла
    def rematerialize(self, new_material):
        self.material = new_material
        self.void_test()

    # | Быстрое обновление параметров тайла
    def replace(self, new_material=False, new_shape=False, new_cover=False):
        if new_material is not False:
            self.material = new_material
        if new_shape is not False:
            self.shape = new_shape
        if new_cover is not False:
            self.cover = new_cover
        self.void_test()

    # | Высчитывание яркости тайла
    def get_illumination(self):
        return max([self.material.illumination,
                    self.light,
                    self.cover.illumination if self.cover else 0])

    # | Генерация текстуры частицы тайла в соответствии с цветами его покрывающего и базового материала
    def get_particle(self):
        if self.cover:
            a = [self.material.color] * 3 + [self.cover.color] * 3
            random.shuffle(a)
            return HexagonTextureRenderer(1, a)
        return HexagonTextureRenderer(1, [self.material.color] * 6)

    # | Звучание тайла в соответствии со звуками его покрывающего и базового материала
    def sound(self):
        sound(self.material.sound)
        if self.cover:
            sound(self.cover.sound)

# Класс чанка
class Chunk():
    # | Инициализация принимает координаты чанка, родительский мир, и имя файла,
    # | содержащего информацию о чанке, если такой есть.
    def __init__(self, coords, world, container=None):
        self.world = world
        self.coords = coords
        self.last_light_update = -1
        self.tiles = [[None] * CHUNK_SIZE
                      for _ in range(CHUNK_SIZE)]
        if container is None:
            self.fresh_init()
        else:
            self.texture_changed = False
            self.read(container)
        self.update_light()

    # | Инициализация чанка с нуля, выбор биома и генерация ландшафта.
    def fresh_init(self):
        self.climate = keep_in_range(int(perlin1d(world.seed + 1005, self.coords.x / (3 * CHUNK_SIZE)) * 1000),
                                     0,
                                     255)
        self.biome = [biome for biome in BIOMES if self.climate in biome.climates][0]
        self.texture_changed = True
        self.generate()        

    # | Загрузка данных чанка в файл
    def dump(self, filename):
        with open(filename, 'wb') as save:
            save.write(bytes([self.climate]))
            for line in self.tiles:
                for tile in line:
                    save.write(bytes([tile.material.id,
                                      tile.cover.id if tile.cover else 0,
                                      tile.shape.id]))
                    
    # | Чтение чанка из файла
    def read(self, filename):
        try:
            with open(filename, mode='rb') as save:
                self.climate = ord(save.read(1))
                self.biome = [biome for biome in BIOMES if self.climate in biome.climates][0]
                for y in range(CHUNK_SIZE):
                    for x in range(CHUNK_SIZE):
                        self.tiles[y][x] = Tile(chunk=self,
                                                material=MATERIALS[ord(save.read(1))],
                                                cover=MATERIALS[ord(save.read(1))],
                                                shape=SHAPES[ord(save.read(1))])
        except Exception as e:
            # || В случае какой либо ошибки, связанной с входными данными, файл чанка
            # || считается повреждённым и чанк генерируется с нуля
            print(f'Corrupted chunk file met! {type(e).__name__} occured while reading', filename)
            self.fresh_init()

    # | Генерация ландшафта
    def generate(self):
        cx, cy = self.coords
        left_biome = predict_biome((cx - 1, cy), self.world)

        grass_layer = set()

        for x in range(CHUNK_SIZE):
            gx = cx * CHUNK_SIZE + x
            level = perlin1d(self.world.seed, gx / 100) * 100 + 100
            for y in range(CHUNK_SIZE):
                gy = cy * CHUNK_SIZE + y
                if level - 0.5 * (gx % 2) > gy:
                    self.tiles[y][x] = Tile(self, AIR, VOID)
                    continue
                cave_pocket = perlin(self.world.seed,
                              Vector2(gx / 32, (gy + 0.5 * (gx % 2)) / 32))
                cave_tunnel = perlin(self.world.seed * 4,
                              Vector2(gx / 42, (gy + 0.5 * (gx % 2)) / 42))
                mineral_distribution = perlin(self.world.seed * 8,
                              Vector2(gx / 128, (gy + 0.5 * (gx % 2)) / 128))
                cave_pocket += ((gy - 350) * 0.001) if gy <= 350 else (gy - 350) * 0.000025
                cave_tunnel += ((gy - 350) * 0.001) if gy <= 350 else (gy - 350) * 0.000025
                
                if level - 0.5 * (gx % 2) + level + 30 <= gy:
                    self.tiles[y][x] = Tile(self, STONE, SOLID)
                else:
                    self.tiles[y][x] = Tile(self, self.biome.soil, SOLID)
                    if left_biome != self.biome:
                        if int(perlin1d(self.world.seed + cx, gy / 8) * 30) >= x:
                            self.tiles[y][x].rematerialize(left_biome.soil)
                if cave_pocket > 0.33 or 0.2 < cave_tunnel < 0.3:

                    self.tiles[y][x].reshape(WALL)

                if 0 <= y - 1 < CHUNK_SIZE and \
                   self.tiles[y - 1][x].shape.is_empty and \
                   not self.tiles[y][x].shape.is_empty:
                    grass_layer.add(self.tiles[y][x])

        for tile in grass_layer:
            if not tile.material.fertile:
                return
            tile.cover = self.biome.flora
            if tile.cover:
                tile.reshape(MOSSED)

    # | Хэш чанка, используется как ключ для хранения текстуры в кэше текстур
    def __hash__(self):
        return hash(self.coords)

    # | Отрисовка тайлов чанка
    def render(self, scale):
        screen = pygame.Surface((CHUNK_SIZE * scale * 4,
                                 int((CHUNK_SIZE * 2 * SQRT_3 + 2) * scale)),
                                pygame.SRCALPHA, 32)
        screen.convert_alpha()

        blits = []
        for y in range(CHUNK_SIZE):
            for x in range(CHUNK_SIZE):
                tile = self.tiles[y][x]
                if tile.shape == 0:
                    continue
                shift = (x % 2) * 0.5

                blits.append((tile.render(scale), (int(x * scale * 3),
                                                   int((y + shift) * scale * SQRT_3 * 2))))
        screen.blits(tuple(blits))
        return screen

    # | Рисование чанка на заданной плоскости.
    def draw(self, coords, scale, screen):
        if True not in [False in [tile.material == AIR for tile in line] for line in self.tiles]:
            return
        key = hash(self)
        if self.texture_changed or key not in TEXTURE_CACHE:
            TEXTURE_CACHE[key] = self.render(scale)
            self.texture_changed = False

        screen.blit(TEXTURE_CACHE[key], coords)

    # | Получение тайла по его относительным координатам в чанке. В случае выхода
    # | за границы чанка обращается к загруженным соседям.
    def get_tile(self, coords: Vector2):
        cx, cy = coords.tuple()
        if 0 <= cx < CHUNK_SIZE and 0 <= cy < CHUNK_SIZE:
            return self.tiles[cy][cx]
        else:
            chx, chy = self.coords.tuple()
            try:
                return self.world.get_chunk(Vector2(chx + cx // CHUNK_SIZE,
                                                      chy + cy // CHUNK_SIZE),
                                            True).get_tile(Vector2(cx % CHUNK_SIZE,
                                                                     cy % CHUNK_SIZE))
            except:
                return None

    # | Объявление изменения текстуры чанка для перерисовки её в соответствии с
    # | обновлёнными данными чанка в следующем кадре.
    def redraw(self):
        self.texture_changed = True

    # | Обновление освещения тайлов внутри чанка (ужас)
    def update_light(self, continue_reaction=True):
        if TICK == self.last_light_update:
            return
        prev = None
        self.last_light_update = TICK
        while True:
            light_map = [[self.get_tile(Vector2(x, y)).get_illumination() if self.get_tile(Vector2(x, y)) else 0 for x in range(-1, CHUNK_SIZE + 1)] for y in range(-1, CHUNK_SIZE + 1)]
            if light_map == prev:
                break
            self.texture_changed = True
            for y in range(CHUNK_SIZE):
                for x in range(CHUNK_SIZE):
                    shift = (x % 2) * 2
                    self.tiles[y][x].light = max([light_map[y + 1][x + 1],
                                                  light_map[y][x + 1],
                                                  light_map[y + 2][x + 1],
                                                  light_map[y + 1][x],
                                                  light_map[y + 1][x + 2],
                                                  light_map[y + shift][x],
                                                  light_map[y + shift][x + 2],
                                                  ]) - self.tiles[y][x].light_res
            prev = light_map
        if continue_reaction:
            w, c = self.world, self.coords
            cs = filter(bool, [w.get_chunk(c + Vector2(1, 0), True),
                               w.get_chunk(c + Vector2(-1, 0), True),
                               w.get_chunk(c + Vector2(0, 1), True),
                               w.get_chunk(c + Vector2(0, -1), True)])

            for c in cs:
                c.update_light(False)

# Предсказание биома для абстрактного чанка на заданных координатах
def predict_biome(chunk_coords, world):
    cx, cy = chunk_coords
    climate = keep_in_range(int(perlin1d(world.seed + 1005, cx / (3 * CHUNK_SIZE)) * 1000),
                            0,
                            255)
    return [biome for biome in BIOMES if climate in biome.climates][0]

# Класс мира
class World:
    # | Инициализация принимает имя мира
    def __init__(self, name):
        self.name = name
        self.cameras = []
        self.loaded_chunks = {}

        # || Если мира с данным именем не существует (а именно если нет папки с именем мира в которой
        # || содержится meta файла), создаётся папка мира с необходимой структурой и объявляется
        # || meta файл. Meta файл содержит информацию о сиде мира, об изученных игроком
        # || материалах и о положении камеры.
        if not os.path.exists('saves/' + name + '/meta.txt'):
            os.makedirs('saves/' + name + '/map/')
            from random import randint
            self.seed = randint(0, 99999999)            
            with open('saves/' + name + '/meta.txt', mode='w') as head:
                head.write(str(self.seed) + '\n')
                head.write('500;100\n')
                head.write(chr(GRASS.id))
        else:
            with open('saves/' + name + '/meta.txt', mode='r') as head:
                self.seed = int(head.readline())

    # | Загрузка заданной области чанков
    def load_zone(self, center: Vector2, radius):
        center = center.map(lambda x: int(x) // CHUNK_SIZE)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius):
                coords = center + Vector2(dx, dy)
                cpath = f'saves/{self.name}/map/{coords[0]};{coords[1]}.chunk'
                if coords in self.loaded_chunks:
                    continue
                if os.path.isfile(cpath):
                    container = cpath
                else:
                    container = None
                self.loaded_chunks[coords] = Chunk(coords, self, container)

    # | Сохранение всех загруженных чанков в файлы
    def save_all(self):
        for coords, chunk in self.loaded_chunks.items():
            chunk.dump(f'saves/{self.name}/map/{coords[0]};{coords[1]}.chunk')

    # | Выгрузка чанка с последующим сохранением его в файл и удалением текстуры чанка из кэша текстур
    def unload(self, coords):
        key = hash(coords)
        self.loaded_chunks[coords].dump(f'saves/{self.name}/map/{coords[0]};{coords[1]}.chunk')
        del self.loaded_chunks[coords]
        if key in TEXTURE_CACHE:
            del TEXTURE_CACHE[key]

    # | Выгрузка области чанков
    def unload_out_of_zone(self, center: Vector2, radius):
        center = center // CHUNK_SIZE
        keys = list(self.loaded_chunks.keys())
        for key in keys:
            d = center - key
            if d.x > radius or d.y > radius:
                self.unload(key)

    # | Отрисовка области чанков
    def draw_zone(self, camera, view, screen, screen_coords):
        center = camera.center.map(lambda x: int(x) // CHUNK_SIZE)
        xradius, yradius = view
        scale = camera.scale
        s = Vector2(*screen_coords)
        for dy in range(-yradius, yradius):
            for dx in range(-xradius, xradius):
                if center + Vector2(dx, dy) not in self.loaded_chunks:
                    continue
                self.loaded_chunks[center + Vector2(dx, dy)].draw((s.x + dx * scale * CHUNK_SIZE * 3,
                                                                     s.y + dy * scale * SQRT_3 * CHUNK_SIZE * 2),
                                                                    scale,
                                                                    screen)

    # | Получение чанка по координатам, будь то тайловым координатам или чанковым
    def get_chunk(self, coords: Vector2, chunk_format=False):
        if chunk_format:
            if coords not in self.loaded_chunks:
                return None
            return self.loaded_chunks[coords]

        chunk = coords // CHUNK_SIZE
        return self.get_chunk(chunk, True)

    # | Получение тайла по координатам
    def get_tile(self, coords: Vector2):
        cx, cy = coords.tuple()
        chunk = self.get_chunk(coords)
        if chunk is None:
            return None
        return chunk.tiles[cy % CHUNK_SIZE][cx % CHUNK_SIZE]

    # | Замена параметров тайла на данных координатах с частицами и звуковым сопровождением.
    def replace_tile(self, coords: Vector2, new_material=False, new_shape=False, new_cover=False):
        chunk = self.get_chunk(coords)
        tile = chunk.get_tile(coords % CHUNK_SIZE)
        self.particle_cast(tile.get_particle(), 10, coords.align())
        tile.sound()
        tile.replace(new_material, new_shape, new_cover)
        chunk.light_changing = True
        chunk.update_light()
        chunk.redraw()

    # | Конвертация мировых координат в оконные для заданной камеры
    def world_to_window_coords(self, camera, tile_world_coords):
        scale = camera.scale
        h = (scale * SQRT_3) / 2
        cwo = camera.center.map(int)
        cwi = camera.get_render_anchor()
        two = tile_world_coords
        cwi += Vector2((cwo.x % CHUNK_SIZE) * scale * 3, (cwo.y % CHUNK_SIZE) * h * 4)

        tdwo = cwo - two
        tdwi = Vector2(tdwo.x * scale * 3, tdwo.y * h * 4)
        twi = cwi - tdwi
        return twi

    # | Вызов пучка частиц на все следящие за миром камеры
    def particle_cast(self, texture, amount, world_coords, weight=1):
        numbers = range(-10, 10)
        for _ in range(amount):
            for camera in self.cameras:
                Particle(camera, texture.render(camera.scale), world_coords, random.choice(numbers) / 100, random.choice(numbers) / 100, weight)

    # | Конвертация оконных координат в мировые
    def window_to_world_coords(self, camera, tile_window_coords):
        scale = camera.scale
        h = (scale * SQRT_3) / 2
        cwo = camera.center.map(int)
        cwi = camera.get_render_anchor()
        twi = Vector2(*tile_window_coords)
        cwi = cwi + Vector2((cwo.x % CHUNK_SIZE) * scale * 3, (cwo.y % CHUNK_SIZE) * h * 4)

        tdwix = cwi.x - twi.x
        tdwox = tdwix // (scale * 3)
        twox = cwo.x - tdwox

        twi.y -= ((twox - 1) % 2) * h * 2
        tdwiy = cwi.y - twi.y
        tdwoy = tdwiy // (h * 4)
        twoy = cwo.y - tdwoy

        return Vector2(int(twox) - 1, int(twoy) - 1)

# Класс камеры
class Camera:
    def __init__(self):
        self.scale = SCALE
        self.center = Vector2(0.0, 0.0)
        self.window_position = (500, 500)
        self.particle_layer = pygame.sprite.Group()
        
    # | Фокусировка камеры на цели. В качестве целей могут уточняться мир или точка.
    def focus(self, *targets):
        for target in targets:
            if isinstance(target, World):
                self.world = target
                target.cameras.append(self)
            elif isinstance(target, tuple):
                self.center = Vector2(*target)

    # | Движение камеры на заданный вектор
    def move(self, x, y):
        self.center += Vector2(x, y)
        if self.center.x <= CHUNK_SIZE or self.center.y <= CHUNK_SIZE:
            self.center -= Vector2(x, 0)
        if self.center.y <= CHUNK_SIZE:
            self.center -= Vector2(0, y)

    # | Отрисовка видимой камере области мира на заданную плоскость
    def render(self, screen, view_distance):
        anchor = self.get_render_anchor()
        self.world.draw_zone(self, (view_distance, view_distance - 1), screen, anchor)
        self.particle_layer.update()
        self.particle_layer.draw(screen)

    # | Установка координат центра изображения камеры для отрисовки на плоскости
    def set_window_position(self, position):
        self.window_position = position

    # | Высчитывание "якоря" рендера - координат центра изображения камеры с учётом
    # | сдвига внутри чанка.
    def get_render_anchor(self):
        position = self.window_position
        return Vector2(position[0] - (self.center.x % CHUNK_SIZE) * SCALE * 3,
                       position[1] - (self.center.y % CHUNK_SIZE) * SCALE * SQRT_3 * 2)

    # | Конвертация мировых координат в оконные
    def apply_coords(self, tile_coords):
        return self.world.world_to_window_coords(self, tile_coords)

# Класс игрока
class Player:
    # | Инициализация принимает мир к которому подключается игрок
    def __init__(self, world: World):
        self.world = world
        self.camera = Camera()
        # || Чтение нужной игроку информации из meta файла мира
        with open('saves/' + world.name + '/meta.txt') as head:
            head.readline()
            self.camera.focus(world, tuple(map(int, head.readline().split(';'))))
            self.available_tiles = list(map(lambda i: MATERIALS[ord(i)], head.readline().split(';')))
        self.inventory_cursor = 0
        self._last_cut = None
        self._cut_iter = 0

    # | Сохранение информации об игроке в meta файл мира
    def save(self):
        head = open('saves/' + self.world.name + '/meta.txt', "r")
        head_prev = head.readlines()
        head_prev[1] = f'{int(self.camera.center.x)};{int(self.camera.center.y)}' + '\n'
        head_prev[2] = ';'.join([chr(mat.id) for mat in self.available_tiles])
        head = open('saves/' + self.world.name + '/meta.txt', "w")
        head.writelines(head_prev)
        head.close()

    # | Разрушение тайла на определённых координатах от лица игрока
    def break_tile(self, coords: Vector2):
        tile = self.world.get_tile(coords)
        if tile.material not in self.available_tiles and tile.material is not AIR:
            self.available_tiles.append(tile.material)
        if tile.cover and tile.cover not in self.available_tiles and tile.cover is not AIR:
            self.available_tiles.append(tile.cover)
        self.world.replace_tile(coords,
                                AIR, VOID, None)

    # | Резка тайла (смена формы) на определённых координатах от лица игрока
    def cut_tile(self, coords: Vector2):
        tile = self.world.get_tile(coords)
        if tile.material is AIR:
            return
        if coords != self._last_cut:
            self._cut_iter = 1
        self._last_cut = coords
        self.world.replace_tile(coords,
                                new_shape=SHAPES[self._cut_iter],
                                new_cover=AIR if 2 in SHAPES[self._cut_iter].texture_template else None)
        self._cut_iter += 1
        self._cut_iter = 1 + (self._cut_iter - 1) % (len(SHAPES) - 1)

    # | Постановка тайла на определённых координатах от лица игрока
    def place_tile(self, coords: Vector2):
        self.world.replace_tile(coords,
                                self.available_tiles[self.inventory_cursor],
                                SOLID, None)

    # | Отрисовка инвенторя игрока на заданной плоскости
    def render_inventory(self, scale, screen):
        screen.fill((133, 133, 133), (0, 0, scale * 26, scale * 4))
        screen.fill((108, 109, 112), (0, scale * 4, scale * 26, scale))
        screen.fill((169, 178, 199), (scale, 0, scale * 4, scale * 4))
        show = self.available_tiles[self.inventory_cursor:self.inventory_cursor + 6]
        for x, obj in enumerate(show):
            key = 'ICON_' + obj.name
            if key not in TEXTURE_CACHE:
                TEXTURE_CACHE[key] = HexagonTextureRenderer(2, [obj.color] * 24).render(scale)
            screen.blit(TEXTURE_CACHE[key], (x * scale * 4 + scale, scale / 4))

    # | Прокрутка инвентаря
    def scroll_inventory(self, direction):
        if 0 <= self.inventory_cursor + direction <= len(self.available_tiles) - 1:
            self.inventory_cursor += direction

    # | Обработчик событий
    def event_handler(self, coords: Vector2, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            is_air = self.world.get_tile(coords).material is AIR
            if event.button == pygame.BUTTON_LEFT:
                if is_air:
                    self.place_tile(coords)
                else:
                    self.break_tile(coords)
            elif event.button == pygame.BUTTON_RIGHT:
                self.cut_tile(coords)
            elif event.button == pygame.BUTTON_WHEELDOWN:
                self.scroll_inventory(-1)
            elif event.button == pygame.BUTTON_WHEELUP:
                self.scroll_inventory(1)

# Загрузка изображения из ассетов в виде плоскости pygame
def load_image(name):
    fullname = os.path.join('assets', name)
    if not os.path.isfile(fullname):
        print(f"Файл с изображением '{fullname}' не найден")
    image = pygame.image.load(fullname)
    return image

# Генератор функции кнопки интерфейса для подключения к определённому миру
def connection_to_world(world):
    def ret(self):
        global running, chosen
        running = False
        chosen = World(world)
    return ret

# Функция кнопки интерфейса для создания нового мира и подключения к нему
def connection_to_new_world(self):
    worlds = [f for f in os.listdir('saves/') if os.path.isfile('saves/' + f + '/meta.txt')]
    i = 1
    while True:
        name = 'World ' + str(i)
        if name not in worlds:
            global running, chosen
            running = False
            chosen = World(name)
            return
        i += 1

# Генератор функции кнопки интерфейса для удаления определённого мира
def world_deletion(world):
    def ret(self):
        import shutil
        shutil.rmtree('saves/' + world)
        self.parent.kill()
        self.kill()
    return ret

# Меню выбора мира
def select_world():
    global running, chosen, buttons
    running = True
    chosen = None
    # | Инициализация интерфейса
    buttons = pygame.sprite.Group()
    worlds = [f for f in os.listdir('saves/') if os.path.isfile('saves/' + f + '/meta.txt')]
    button_delete = load_image('button_delete.png')
    for y, world in enumerate(worlds):
        button = Button(world)
        button.rect.y = 200 + y * 190
        button.func = connection_to_world(world)
        delete = Button(None)
        delete.image = button_delete
        delete.rect.y = 200 + y * 190
        delete.rect.x += 550
        delete.parent = button
        delete.func = world_deletion(world)
    new_world = Button('New world')
    new_world.rect.y = 1000
    new_world.rect.x = 1000
    new_world.func = connection_to_new_world
    new_world.scrollable = False

    # | Цикл окна
    while running:
        # || Обработчик событий
        for event in pygame.event.get():
            # ||| События кнопок
            buttons.update(event)
            # ||| Закрытие окна
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()
        # || Рендер интерфейса
        screen.fill((137, 166, 240))
        buttons.draw(screen)
        # || Завершение кадра
        pygame.display.flip()
    # | Закрытие меню
    buttons.empty()
    return chosen

# Функция-заглушка чтобы было что вставить как дефолтное действие для кнопки :P
def _pass(*args, **kwargs):
    pass

# Класс кнопки
class Button(pygame.sprite.Sprite):
    bg = load_image('button.png')
    # | Инициализация принимает название кнопки и по необходимости группы спрайтов для кнопки
    # | Остальные кастомизационные значения задаются прямым изменением атрибутов.
    def __init__(self, name, *group):
            super().__init__(buttons, *group)
            if name:
                self.image = Button.bg.copy()
                self.image.blit(font.render(name, True, pygame.Color('white')),
                                (50, 50))
            else:
                self.image = Button.bg
            self.rect = self.image.get_rect()
            self.rect.x = 100
            self.rect.y = 500
            self.func = _pass
            self.scrollable = True

    # | Обработчик событий
    def update(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            # || Нажатие на кнопку
            if event.button == pygame.BUTTON_LEFT and \
               self.rect.collidepoint(event.pos):
                self.func(self)
                sound('button')
            # || Прокрутка интерфейса колёсиком мыши
            elif self.scrollable and event.button == pygame.BUTTON_WHEELUP:
                self.rect.y += 40
            elif self.scrollable and event.button == pygame.BUTTON_WHEELDOWN:
                self.rect.y -= 40


if __name__ == '__main__':
    if not os.path.exists('saves/'):
        os.makedirs('saves/')
    
    # Инициализация Pygame, окна и шрифта
    pygame.init()
    screen = pygame.display.set_mode((2000, 1200), pygame.RESIZABLE)
    pygame.display.set_caption('Hexagonal sandbox')
    font = pygame.font.Font(None, 60)

    # Открытие меню выбора мира
    world = select_world()

    # Инициализация игры в мире
    pygame.display.set_caption('Hexagonal sandbox — ' + world.name)
    # | Локальные константы
    view_distance = 4
    TICK = 0
    fullscreen_timeout = 0
    debug_screen = False
    music_on = True
    bg_color = (255, 255, 255)
    # | Подключение игрока
    PLAYER = Player(world)
    # Запуск игрового цикла
    clock = pygame.time.Clock()
    pygame.mixer.music.load('assets/song.mp3')
    running = True
    while running:
        # Обновление сведений о размерах окна
        window_width, window_height = screen.get_size()
        SCREEN_BORDER = (0, 0, window_width, window_height)
        PLAYER.camera.set_window_position((window_width // 2, window_height // 2))

        # Загрузка территории вокруг игрока и отгрузка неиспользующихся чанков
        world.load_zone(PLAYER.camera.center, view_distance)
        world.unload_out_of_zone(PLAYER.camera.center, view_distance)

        # Получение тайла на который наведён курсок
        selected_coords = world.window_to_world_coords(PLAYER.camera, pygame.mouse.get_pos())

        # Проигрывание музыки
        if music_on and not pygame.mixer.music.get_busy():
            if TICK % 5000 == random.randint(0, 5000):
                pygame.mixer.music.play()
        
        # Обработчик событий
        # | Импульсные события
        for event in pygame.event.get():
            PLAYER.event_handler(selected_coords, event)
            # || Выход из игры с сохранением информации о мире
            if event.type == pygame.QUIT:
                world.save_all()
                PLAYER.save()
                running = False
            # || Функциональные клавиши
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F11 and not fullscreen_timeout:
                    pygame.display.toggle_fullscreen()
                    fullscreen_timeout = 100
                elif event.key == pygame.K_F3:
                    debug_screen = not debug_screen
                elif event.key == pygame.K_F8:
                    music_on = not music_on
                    if not music_on:
                        pygame.mixer.music.fadeout(500)
        # | Зажатые клавиши (движение камеры)
        keys = pygame.key.get_pressed()
        PLAYER.camera.move((int(keys[pygame.K_a]) * -1 + int(keys[pygame.K_d])) * SPEED,
                           (int(keys[pygame.K_w]) * -1 + int(keys[pygame.K_s])) * SPEED)

        # Рендер
        # | Плавная заливка фона в соответствии с биомом в котором находится камера
        bg_color = smooth_color(bg_color, world.get_chunk(PLAYER.camera.center).biome.skycolor, 1)
        screen.fill(bg_color)
        # | Отображение видимой камере территории
        PLAYER.camera.render(screen, view_distance)
        # | Обводка выбранного тайла
        tile_window_coords = world.world_to_window_coords(PLAYER.camera, selected_coords.tile_shift()) + \
            Vector2(SCALE * 2, SCALE * SQRT_3)
        points = get_points(SCALE)
        cursor_color = 191 + abs(32 - TICK // 3 % 64)
        pygame.draw.polygon(screen,
                            (cursor_color, cursor_color, cursor_color),
                            [(points[0][0] + tile_window_coords.x, points[0][1] + tile_window_coords.y),
                             (points[1][0] + tile_window_coords.x, points[1][1] + tile_window_coords.y),
                             (points[2][0] + tile_window_coords.x, points[2][1] + tile_window_coords.y),
                             (points[3][0] + tile_window_coords.x, points[3][1] + tile_window_coords.y),
                             (points[4][0] + tile_window_coords.x, points[4][1] + tile_window_coords.y),
                             (points[5][0] + tile_window_coords.x, points[5][1] + tile_window_coords.y)],
                            4)
        # | Отрисовка содержимого инвентаря игрока
        PLAYER.render_inventory(20, screen)
        # | Отрисовка экрана отладки, если тот включён
        if debug_screen:
            screen.blit(font.render(f'FPS: {int(clock.get_fps())}', True, pygame.Color('white')),
                        (50, 140))
            screen.blit(font.render(f'Chunks Loaded: {len(world.loaded_chunks)}', True, pygame.Color('white')),
                        (50, 180))
            screen.blit(font.render(f'X/Y: {PLAYER.camera.center}', True, pygame.Color('white')),
                        (50, 220))
            screen.blit(font.render(f'Textures Loaded: {len(TEXTURE_CACHE)}', True, pygame.Color('white')),
                        (50, 260))
            screen.blit(font.render(f'VGC: {len(VGCACHE)}', True, pygame.Color('white')),
                        (50, 300))

        # Завершение кадра
        pygame.display.flip()
        clock.tick(60)
        TICK += 1
        fullscreen_timeout -= 1 if fullscreen_timeout else 0

    pygame.quit()