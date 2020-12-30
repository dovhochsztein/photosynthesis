import hexutil
from collections import namedtuple
from copy import *

class Goes_In_Hex():
    def __init__(self, value=None, text=[]):
        self.value = value
        self.text = text
    def __str__(self):
        return str(self.name)

# class blocked(Goes_In_Hex):
#     def __init__(self, value, text=None):
#         super.__init__(self, value, text)

class Hex():
    """Hexagonal grid element (cubic coordinates)
    sign convention based on 'odd-q' https://www.redblobgames.com/grids/hexagons/

     /+y  \ \      /
    /    +x\ \____/
    \      / /    \
     \+z__/ /      \
     /    \ \      /
    /      \ \____/
    \      / /    \
     \____/ /      \
    """
    def __init__(self, coordinates=(0, 0, 0), obj=None):
        # if len(coordinates) == 2:
        #     self.rectangular_coordinates = coordinates
        # else:
        self.cubic_coordinates = coordinates
        self.rectangular_coordinates = (coordinates[0], coordinates[2] + (coordinates[0] - (coordinates[0] % 2)) / 2)
        # self.radius = max([abs(c) for c in coordinates])
        self.radius = None
        self.obj = obj
        # if obj and hasattr(obj, 'text'):
        #     self.text = obj.text
        # else:
        # self.text = []
        self.text = str(obj).split('\n')

    def get_radius(self):
        if self.radius is None:
            self.radius = max([abs(c) for c in self.cubic_coordinates])
        return self.radius

    def __deepcopy__(self, memodict={}):
        new = Hex(self.cubic_coordinates, self.obj)
        return new

    def __eq__(self, other):
        return self.obj == other.obj

    def update_rectangular_coordinates(self, origin):
        # self.rectangular_coordinates = (self.rectangular_coordinates[0] - origin[0], int(self.rectangular_coordinates[1] - origin[1] + self.rectangular_coordinates[0] % 2 - origin[0] % 2))
        self.rectangular_coordinates = (self.rectangular_coordinates[0] - origin[0], int(self.rectangular_coordinates[1] - origin[1] + (self.rectangular_coordinates[0] % 2 and origin[0] % 2)))

    def change(self, obj):
        self.obj = obj
        if obj and hasattr(obj, 'text'):
            self.text = obj.text
        else:
            self.text = []
            self.text = str(obj).split('\n')

    def __str__(self):
        return str(self.obj)


class Grid:
    """heagonal grid consisting of a list of hexagonal elements"""
    def __init__(self, element_list, width=5, height=7, size=2):
        self.element_list = element_list
        # origin = [0, 0]
        # for element in element_list:
        #     if element.rectangular_coordinates[0] < origin[0]:
        #         origin[0] = element.rectangular_coordinates[0]
        #     if element.rectangular_coordinates[1] - (origin[0] % 2) < origin[1]:# - (origin[0] % 2):
        #         origin[1] = int(element.rectangular_coordinates[1] - (origin[0] % 2))# - (origin[0] % 2)
        # for element in self.element_list:
        #     element.update_rectangular_coordinates(origin)
        #     if element.rectangular_coordinates[0] + 1 > width:
        #         width = int(element.rectangular_coordinates[0]) + 1
        #     if element.rectangular_coordinates[1] + 2 > height:
        #         height = int(element.rectangular_coordinates[1]) + 2
        self.origin = None
        self.width = width
        self.height = height
        self.size = size
        self.element_dict = {element.cubic_coordinates: element for element in self.element_list}

    def __deepcopy__(self, memodict={}):
        new = Grid([deepcopy(element) for element in self.element_list])
        # new.element_list = [deepcopy(element) for element in self.element_list]
        new.origin = self.origin
        new.width = self.width
        new.height = self.height
        new.element_dict = {element.cubic_coordinates: element for element in new.element_list}
        return new

    def __eq__(self, other):
        if len(self.element_list) != len(other.element_list):
            return False
        for coordinates in self.element_dict:
            if coordinates not in other.element_dict or self.element_dict[coordinates] != other.element_dict[coordinates]:
                return False

    def add_hex(self, hex):
        if hex.cubic_coordinates in self.element_dict:
            raise NameError('location already taken')
        hex.update_rectangular_coordinates(self.origin) #fix?
        self.element_list.append(hex)
        self.element_dict[hex.cubic_coordinates] = hex
        if hex.rectangular_coordinates[0] + 1 > self.width:
            self.width = int(element.rectangular_coordinates[0]) + 1
        if element.rectangular_coordinates[1] + 2 > self.height:
            self.height = int(element.rectangular_coordinates[1]) + 2

    def where(self, obj, comparison=None):
        output = []
        for element in self.element_list:
            if comparison == None:
                if element.obj == obj:
                    output.append(element.cubic_coordinates)
            else:
                if comparison(element.obj, obj):
                    output.append(element.cubic_coordinates)
        return output

    def __getitem__(self, key):
        return self.element_dict[key]

    def __str__(self):
        if self.origin is None:
            origin = [0, 0]
            for element in element_list:
                if element.rectangular_coordinates[0] < origin[0]:
                    origin[0] = element.rectangular_coordinates[0]
                if element.rectangular_coordinates[1] - (origin[0] % 2) < origin[1]:  # - (origin[0] % 2):
                    origin[1] = int(element.rectangular_coordinates[1] - (origin[0] % 2))  # - (origin[0] % 2)
            for element in self.element_list:
                element.update_rectangular_coordinates(origin)
                if element.rectangular_coordinates[0] + 1 > width:
                    width = int(element.rectangular_coordinates[0]) + 1
                if element.rectangular_coordinates[1] + 2 > height:
                    height = int(element.rectangular_coordinates[1]) + 2
            self.origin = origin
        return generate_visual_grid(self.element_list, self.width, self.height, size=self.size)
    pass


def translation(cubic_coordinates, direction, distance):
    """
    Gives the cubic coordinates of the hex gotten by translating a distance in a direction
    direction = 0 is downward in the grid, increasing rotates counter-clockwise
    """
    x, y, z = cubic_coordinates
    if direction == 0 or direction == 1:
        y -= distance
    if direction == 1 or direction == 2:
        x += distance
    if direction == 2 or direction == 3:
        z -= distance
    if direction == 3 or direction == 4:
        y += distance
    if direction == 4 or direction == 5:
        x -= distance
    if direction == 5 or direction == 0:
        z += distance
    return (x, y, z)


def distance(cubic_coordinates_1, cubic_coordinates_2):
    delta_x = abs(cubic_coordinates_1[0] - cubic_coordinates_2[0])
    delta_y = abs(cubic_coordinates_1[1] - cubic_coordinates_2[1])
    delta_z = abs(cubic_coordinates_1[2] - cubic_coordinates_2[2])
    return sum([delta_x, delta_y, delta_z]) - max(delta_x, delta_y, delta_z)


def generate_visual_grid(element_list, width, height, size=2):
    building_blocks = []
    size_y = int(round(size * 1.7)) + 2
    for ii in range(size):
        new_line = ' ' * (size - ii - 1) + "/" + ' ' * (size_y + ii *2) + "\\" + ' ' * (size - ii - 1)
        building_blocks.append(new_line)
    for ii in reversed(range(size)):
        if ii == 0:
            new_line = ' ' * (size - ii - 1) + "\\" + '_' * (size_y + ii * 2) + "/" + ' ' * (size - ii - 1)
        else:
            new_line = ' ' * (size - ii - 1) + "\\" + ' ' * (size_y + ii *2) + "/" + ' ' * (size - ii - 1)
        building_blocks.append(new_line)
    top_line_building_block = ' ' * (size)+ '_' * (size_y) + ' ' * (size)
    empty_building_block = ' ' * (size_y + 2 * size)
    lines = ['']
    for jj in range(width):
        lines[0] += empty_building_block if jj % 2 else top_line_building_block
    for ii in range(height * size * 2):
        new_line = ''
        for jj in range(width):
            new_line += building_blocks[(ii + size * (jj % 2)) % (size*2)]
        lines.append(new_line)
    for element in element_list:
        x_center = int((size_y + 2 * size) * (element.rectangular_coordinates[0] + 0.5))
        y = int(size * 2 * element.rectangular_coordinates[1] + 1 + (element.rectangular_coordinates[0] % 2 + 1) * size - int((len(element.text) + 1)/ 2))
        for ii, chars in enumerate(element.text):
            if ii >= size * 2 - 1:
                break
            space = size_y + 2 * min(ii, abs(ii + 1 - size * 2))
            if space < len(chars):
                chars = chars[0:space]
            x = x_center - int(len(chars)/2)
            lines[y + ii] = lines[y + ii][0: x] + chars + lines[y + ii][x + len(chars):]
    output = '\n'.join(lines)
    return output


def generate_radial_hex_array(radius, default_obj=None):
    """generates an empty array with given radius (radius = 0 gives a single hex)
    """
    # if type(default_obj) != Goes_In_Hex:
    #     default_obj = Goes_In_Hex(value=default_obj, text=[str(default_obj)])
    element_list = [Hex((0, 0, 0), default_obj)]
    for xx in range(-radius, radius + 1):
        for yy in range(-radius, radius + 1):
            if xx == 0 and yy == 0:
                continue
            zz = - xx - yy
            if abs(zz) > radius:
                continue
            element_list.append(Hex((xx, yy, zz), default_obj))
    grid = Grid(element_list)
    return grid


def hexwise_operation(grids, operation):
    new_hex = deepcopy(grids[0])
    for coordinates in grids[0].element_dict:
        if type(grids) == list or type(grids) == tuple:
            if len(grids) == 1:
                new_hex[coordinates].change(operation(grids[0]))
            elif len(grids) > 1:
                new_hex[coordinates].change(operation([grid[coordinates] for grid in grids]))
        else:
            new_hex[coordinates].change(operation(grids))
    return new_hex


if __name__ == '__main__':

    # A = Hex((1, 2), Goes_There('First', ['the', 'quick']))
    # B = Hex((3, 4), Goes_There('Second', ['brown', 'fox', 'jumps']))
    # C = Hex((2, 1), Goes_There('Third', ['over']))
    # D = Hex((4, 3), Goes_There('Fourth', ['the', 'lazy', 'dog', 'end']))
    #
    # A = Hex((1, -3, 2), Goes_In_Hex('First', ['the', 'quick']))
    # B = Hex((3, -6, 3), Goes_In_Hex('Second', ['brown', 'fox', 'jumps']))
    # C = Hex((2, -2, 0), Goes_In_Hex('Third', ['over']))
    # D = Hex((4, -5, 1), Goes_In_Hex('Fourth', ['the', 'lazy', 'dog', 'end']))
    # element_list = [A, B, C, D]
    # G = Grid(element_list, size=2)
    # print(G)

    A = Hex((0, 1, -1), Goes_In_Hex('First', ['the', 'quick']))
    B = Hex((1, -1, 0), Goes_In_Hex('Second', ['brown', 'fox', 'jumps']))
    C = Hex((0, -1, 1), Goes_In_Hex('Third', ['over']))
    D = Hex((-1, 1, 0), Goes_In_Hex('Fourth', ['the', 'lazy', 'dog', 'end']))
    E = Hex((1, 0, -1), Goes_In_Hex('Third', ['more', 'stuff']))
    F = Hex((-1, 0, 1), Goes_In_Hex('Fourth', ['he', 'not', 'name']))

    A = Hex((0, 1, -1), 'the quick')
    B = Hex((1, -1, 0), 'brown fox jumps')
    C = Hex((0, -1, 1), 'over')
    D = Hex((-1, 1, 0), 'the lazy dog end')
    E = Hex((1, 0, -1), 'more stuff')
    F = Hex((-1, 0, 1), 'he not name')
    element_list = [A, B, C, D, E, F]
    G = Grid(element_list, size=2)
    g = generate_radial_hex_array(1)
    g = generate_radial_hex_array(3)
    h = generate_radial_hex_array(3, False)
    h[(3, -2, -1)].change(True)
    h.where(True)
    print(h)
    for element in g.element_list:
        # element.obj.text = ['rad =', str(element.radius)]
        element.text = ['rad =', str(element.radius)]
    print(g)
    print(G)
