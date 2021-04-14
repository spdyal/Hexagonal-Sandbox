# Модуль, в котором собраны различные вспомогательные функции и классы.

from math import sqrt
SQRT_3 = sqrt(3)

# Подгон трёх значений под соответствие интервалу от 0 до 255
def stabilize_color(*color):
    return tuple([keep_in_range(el, 0, 255) for el in color])

# Случайное отклонение цвета
def variate_color(color, bounds):
    from random import randint
    from pygame import Color
    def a(c):
        c += randint(*bounds)
        if c < 0:
            c = 0
        elif c > 255:
            c = 255
        return c
    color = map(a, tuple(color))
    return Color(*color)

# Интерполяция двух цветов
def smooth_color(source_color, dest_color, step):
    x = enumerate(tuple(source_color))
    y = tuple(dest_color)
    def a(inp):
        ind, c = inp
        if c > y[ind]:
            c -= step
        elif c < y[ind]:
            c += step
        return c
    return tuple(map(a, x))

# Получение координат вершин шестиугольника с заданной стороной относительно центра
def get_points(scale):
    h = (scale * SQRT_3) / 2
    return [(0 - scale, 0 - h * 2),
            (scale, 0 - h * 2),
            (scale * 2, 0),
            (scale, h * 2),
            (0 - scale, h * 2),
            (0 - scale * 2, 0)]

# Псевдослучайное число по сиду
def musrand(seed):
    a = int(seed)
    b = a + 1
    for _ in range(10):
        a = (a * 578194587349012378941734 + 174290425205728957) // 100000000
        b = (b * 578194587349012378941734 + 174290425205728957) // 100000000
    a = (a % 671049582825) / 671049582825
    b = (b % 671049582825) / 671049582825
    point = seed - int(seed)
    return a + ((b - a) * point ** 3 * (point * (point * 6 - 15) + 10))

# Подгон значения в интервал
def keep_in_range(num, min, max):
    if num > max:
        return max
    if num < min:
        return min
    return num

# Псевдослучайное целое число из интервала по сиду
def randint(seed, min, max):
    r = max - min
    return min + int(musrand(seed) * 10 ** len(str(r))) % r

# Класс двумерного вектора
class Vector2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # | Кортеж из координат
    def tuple(self):
        return (self.x, self.y)

    # | Выравнивание по центру предполагаемого шестиугольника
    def align(self):
        return self.int() + Vector2(2 / 3, 0.5)

    # | Переопределение операторов
    def __add__(self, other):
        if isinstance(other, Vector2):
            return Vector2(self.x + other.x, self.y + other.y)
        elif isinstance(other, int) or isinstance(other, float):
            return Vector2(self.x + other, self.y + other)
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Vector2):
            return Vector2(self.x - other.x, self.y - other.y)
        elif isinstance(other, int) or isinstance(other, float):
            return Vector2(self.x - other, self.y - other)
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Vector2):
            return Vector2(self.x * other.x, self.y * other.y)
        elif isinstance(other, int) or isinstance(other, float):
            return Vector2(self.x * other, self.y * other)
        else:
            return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, Vector2):
            return Vector2(self.x / other.x, self.y / other.y)
        elif isinstance(other, int) or isinstance(other, float):
            return Vector2(self.x / other, self.y / other)
        else:
            return NotImplemented

    def __floordiv__(self, other):
        if isinstance(other, Vector2):
            return Vector2(self.x // other.x, self.y // other.y)
        elif isinstance(other, int) or isinstance(other, float):
            return Vector2(self.x // other, self.y // other)
        else:
            return NotImplemented

    def __mod__(self, other):
        if isinstance(other, Vector2):
            return Vector2(self.x % other.x, self.y % other.y)
        elif isinstance(other, int) or isinstance(other, float):
            return Vector2(self.x % other, self.y % other)
        else:
            return NotImplemented

    def __iadd__(self, other):
        self = self + other
        return self

    def __isub__(self, other):
        self = self - other
        return self

    def __imul__(self, other):
        self = self * other
        return self

    def __itruediv__(self, other):
        self = self / other
        return self

    def __ifloordiv__(self, other):
        self = self // other
        return self    

    # | Смещение столбца для корректной отрисовки шестиугольной сетки
    def tile_shift(self):
        return Vector2(self.x, self.y + (self.x % 2) * 0.5)

    # | Переопределение оператора ==
    def __eq__(self, other):
        return isinstance(other, Vector2) and self.tuple() == other.tuple()
    
    # | Округление вектора
    def round(self):
        return Vector2(round(self.x), round(self.y))

    # | Вектор из целых частей координат данного
    def int(self):
        return Vector2(int(self.x), int(self.y))

    # | Перенаправление итерации на кортеж
    def __iter__(self):
        return self.tuple().__iter__()

    # | Применение функции к каждой точке
    def map(self, f):
        return Vector2(f(self.x), f(self.y))

    # | Хэш вектора
    def __hash__(self):
        return hash(self.tuple())

    # | Строковое представление вектора
    def __str__(self):
        sx, sy = str(self.x), str(self.y)
        sx = sx[:sx.find('.') + 3]
        sy = sy[:sy.find('.') + 3]
        return f'({sx};{sy})'

    # | Репродукция вектора
    def __repr__(self):
        return f'Vector2({self.x}, {self.y})'

    # | Перенаправление индексации на кортеж
    def __getitem__(self, key):
        return self.tuple()[key]

    # | Склярное произведение двух векторов
    def dot_product(self, other):
        return self.x * other.x + self.y * other.y

    # | Длина вектора 
    def length(self):
        from math import sqrt
        return sqrt(self.x ** 2 + self.y ** 2)

    # | Единичный вектор с тем же направлением
    def unit(self):
        return self / self.length()


# Высчитывание шума перлина.
# Основано на https://gist.github.com/eevee/26f547457522755cb1fb8739d0ea89a1
# | Кэш градиентов
VGCACHE = {}
# | Генерация градиентов
def generate_vector_gradient(seed, x, y, remember=True):
    gradients = [Vector2(1, 0),
                 Vector2(-1, 0),
                 Vector2(0, 1),
                 Vector2(0, -1),
                 Vector2(0.707, 0.707),
                 Vector2(-0.707, 0.707),
                 Vector2(0.707, -0.707),
                 Vector2(-0.707, -0.707)]
    if len(VGCACHE) >= 100:
        VGCACHE.clear()
    if (seed, x, y) in VGCACHE:
        return VGCACHE[(seed, x, y)]
    else:
        v = gradients[randint(x ^ (y + seed), 0, 8)]
        
        if remember:
            VGCACHE[(seed, x, y)] = v
        return v    

# | Линейная интерполяция двух точек
def lerp(a, b, t):
    return a + t * (b - a)

# | Шаг интерполяции для дробного значения
def smoothstep(t):
    return t * t * (3. - 2. * t)

# | Одномерный шум перлина
def perlin1d(seed, x):
    a = int(x)
    b = a + 1
    dots = [(randint(seed ^ a, -100, 100) / 100) * (x - a),
            (randint(seed ^ b, -100, 100) / 100) * (x - b)]
    s = smoothstep(x - a)
    return lerp(*dots, s)

# | Двумерный шум перлина
def perlin(seed, point):
    flx, fly = int(point.x), int(point.y)
    dots = []
    for node in [(flx, fly), (flx + 1, fly),
                 (flx, fly + 1), (flx + 1, fly + 1)]:
        gradient = generate_vector_gradient(seed, *node)
        dots.append((point - Vector2(*node)).dot_product(gradient))
    return lerp(lerp(dots[0], dots[1], smoothstep(point.x - flx)),
                lerp(dots[2], dots[3], smoothstep(point.x - flx)),
                smoothstep(point.y - fly)) + 0.25

# Демонстрация работы одномерного шума перлина
if __name__ == '__main__':
    for y in range(20):
        for x in range(100):
            if int(perlin1d(1, x / 10) * 20) + 10 < y:
                print('@', end='')
            else:
                print('#', end='')
        print()
