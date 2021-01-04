from collections import namedtuple
from copy import *
import numpy as np

class Goes_In_Hex():
    def __init__(self, value=None, text=[]):
        self.value = value
        self.text = text
    def __str__(self):
        return str(self.text)


# todo:
# remove element list,
# fix printing so that it shows only hexes that are included,
# include array slicing,
# fix print text to consider new lines without requiring list of strings
# get rid of changing rectangular coordinates
# add repr to Grid
# consider multiple versions of rectangular coordinates

# class blocked(Goes_In_Hex):
#     def __init__(self, value, text=None):
#         super.__init__(self, value, text)

class Hex:
    """Hexagonal grid element (cubic coordinates)
    sign convention for rectangular coordinates based on 'odd-q' offset coordinates https://www.redblobgames.com/grids/hexagons/

    Cubic:                              Rectangular
       _______                            _______
      /+y     \  \           /           /   +y  \  \           /
     /         \  \         /           /   ^     \  \         /
    /         +x\  \_______/           /    | __\+x\  \_______/
    \           /  /       \           \        /  /  /       \
     \         /  /         \           \         /  /         \
      \+z_____/  /           \           \_______/  /           \
      /       \  \           /           /       \  \           /
     /         \  \         /           /         \  \         /
    /           \  \_______/           /           \  \_______/
    \           /  /       \           \           /  /       \
     \         /  /         \           \         /  /         \
      \_______/  /           \           \_______/  /           \

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

    def __repr__(self):
        return f'Hex({self.cubic_coordinates}, {repr(self.obj)})'


def complete_cubic_coordinates(incomplete_cubic_coordinates):
    cubic_coordinates = (incomplete_cubic_coordinates[0],
                         incomplete_cubic_coordinates[1],
                         -incomplete_cubic_coordinates[0] - incomplete_cubic_coordinates[1])
    return cubic_coordinates


def convert_cubic_coordinates_to_rectangular(cubic_coordinates):
    rectangular_coordinates = (cubic_coordinates[0], int(cubic_coordinates[2] + (cubic_coordinates[0] - (cubic_coordinates[0] % 2)) / 2))
    return rectangular_coordinates


def convert_rectangular_coordinates_to_cubic(rectangular_coordinates):
    cubic_coordinates = (rectangular_coordinates[0], 0, rectangular_coordinates[1] - (rectangular_coordinates[0] - (rectangular_coordinates[0] % 1)) / 2)
    cubic_coordinates = (cubic_coordinates[0], -cubic_coordinates[0] - cubic_coordinates[2], cubic_coordinates[2])
    return cubic_coordinates


class Grid:
    """heagonal grid consisting of a list of hexagonal elements"""
    def __init__(self, element_list, size=3):
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
        rectangular_coordinates_list = [element.rectangular_coordinates for element in element_list]
        x_list = [coord[0] for coord in rectangular_coordinates_list]
        y_list = [coord[1] for coord in rectangular_coordinates_list]
        self.min_x = int(min(x_list))
        self.min_y = int(min(y_list))
        self.max_x = int(max(x_list))
        self.max_y = int(max(y_list))
        self.size = size
        self.element_dict = {element.cubic_coordinates: element for element in self.element_list}

    def __deepcopy__(self, memodict={}):
        new = Grid([deepcopy(element) for element in self.element_list])
        # new.element_list = [deepcopy(element) for element in self.element_list]
        new.origin = self.origin
        new.min_x = self.min_x
        new.min_y = self.min_y
        new.max_x = self.max_x
        new.max_y = self.max_y
        new.element_dict = {element.cubic_coordinates: element for element in new.element_list}
        return new

    def __eq__(self, other):
        if len(self.element_list) != len(other.element_list):
            return False
        for coordinates in self.element_dict:
            if coordinates not in other.element_dict or self.element_dict[coordinates] != other.element_dict[coordinates]:
                return False

    def update_min_max_x_y(self, rectangular_coordinates):
        if rectangular_coordinates[0] < self.min_x:
            self.min_x = int(rectangular_coordinates[0])
        elif rectangular_coordinates[0] > self.max_x:
            self.max_x = int(rectangular_coordinates[0])
        if rectangular_coordinates[1] < self.min_y:
            self.min_y = int(rectangular_coordinates[1])
        elif rectangular_coordinates[1] > self.max_y:
            self.max_y = int(rectangular_coordinates[1])

    def add_hex(self, hex):
        if hex.cubic_coordinates in self.element_dict:
            raise NameError('location already taken')
        hex.update_rectangular_coordinates(self.origin) #fix?
        self.element_list.append(hex)
        self.element_dict[hex.cubic_coordinates] = hex
        self.update_min_max_x_y(hex.rectangular_coordinates)

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

    def get_subgrid(self, coordinate_list):
        #todo: skip indices it cant find
        if len(coordinate_list) == 0:
            return None
        try:
            return Grid([self[coordinates] for coordinates in coordinate_list])
        except:
            raise IndexError

    def __getitem__(self, key):
        if len(key) in [2, 3]:
            print(type(key[0]))
            print(key)
            if isinstance(key[0], slice) or isinstance(key[1], slice):
                starting_index = list()
                ending_index = list()
                if isinstance(key[0], slice):
                    starting_index.append(key[0].start)
                    ending_index.append(key[0].stop)
                else:
                    starting_index.append(key[0])
                    ending_index.append(key[0])
                if isinstance(key[1], slice):
                    starting_index.append(key[1].start)
                    ending_index.append(key[1].stop)
                else:
                    starting_index.append(key[0])
                    ending_index.append(key[0])
                starting_index = complete_cubic_coordinates(starting_index)
                ending_index = complete_cubic_coordinates(ending_index)
                coordinate_list = generate_rhombal_array_indices(starting_index, ending_index)
                return self.get_subgrid(coordinate_list)
            else:
                return self.element_dict[tuple(key)]
        else:
            return None

    def __getslice__(self, i, j, sequence):
        return 'getslice'

    def __setitem__(self, key, value):
        return '__setitem__'

    def __setslice__(self, i, j, sequence):
        return 'setslice'

    def __str__(self):
        # if self.origin is None:
        #     origin = [0, 0]
        #     for element in self.element_list:
        #         if element.rectangular_coordinates[0] < origin[0]:
        #             origin[0] = element.rectangular_coordinates[0]
        #         if element.rectangular_coordinates[1] - (origin[0] % 2) < origin[1]:  # - (origin[0] % 2):
        #             origin[1] = int(element.rectangular_coordinates[1] - (origin[0] % 2))  # - (origin[0] % 2)
        #     for element in self.element_list:
        #         element.update_rectangular_coordinates(origin)
        #         self.update_min_max_x_y(element.rectangular_coordinates)
        #     self.origin = origin
        # return generate_visual_grid(self.element_list, self.width, self.height, size=self.size)
        return generate_visual_grid(self.element_dict, self.min_x, self.min_y, self.max_x, self.max_y, size=self.size)

    def __repr__(self):
        # hex_list = [repr(hex) for hex in self.element_list]
        return f'Grid({self.element_list})'

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


# def generate_visual_grid(element_list, width, height, size=3):
#     ASPECT_RATIO = 1.7
#     building_blocks = []
#     size_y = int(round(size * ASPECT_RATIO)) + 2
#     for ii in range(size):
#         new_line = ' ' * (size - ii - 1) + "/" + ' ' * (size_y + ii *2) + "\\" + ' ' * (size - ii - 1)
#         building_blocks.append(new_line)
#     for ii in reversed(range(size)):
#         if ii == 0:
#             new_line = ' ' * (size - ii - 1) + "\\" + '_' * (size_y + ii * 2) + "/" + ' ' * (size - ii - 1)
#         else:
#             new_line = ' ' * (size - ii - 1) + "\\" + ' ' * (size_y + ii *2) + "/" + ' ' * (size - ii - 1)
#         building_blocks.append(new_line)
#     top_line_building_block = ' ' * (size)+ '_' * (size_y) + ' ' * (size)
#     empty_building_block = ' ' * (size_y + 2 * size)
#     lines = ['']
#     for jj in range(width):
#         lines[0] += empty_building_block if jj % 2 else top_line_building_block
#     for ii in range(height * size * 2):
#         new_line = ''
#         for jj in range(width):
#             new_line += building_blocks[(ii + size * (jj % 2)) % (size*2)]
#         lines.append(new_line)
#     for element in element_list:
#         x_center = int((size_y + 2 * size) * (element.rectangular_coordinates[0] + 0.5))
#         y = int(size * 2 * element.rectangular_coordinates[1] + 1 + (element.rectangular_coordinates[0] % 2 + 1) * size - int((len(element.text) + 1)/ 2))
#         for ii, chars in enumerate(element.text):
#             if ii >= size * 2 - 1:
#                 break
#             space = size_y + 2 * min(ii, abs(ii + 1 - size * 2))
#             if space < len(chars):
#                 chars = chars[0:space]
#             x = x_center - int(len(chars)/2)
#             lines[y + ii] = lines[y + ii][0: x] + chars + lines[y + ii][x + len(chars):]
#     output = '\n'.join(lines)
#     return output


def generate_visual_grid(element_dict, min_x, min_y, max_x, max_y, size=3):
    width = max_x - min_x + 1
    height = max_y - min_y + 2
    overbar = u"\u203E"
    element_rectangular_dict = {hex_obj.rectangular_coordinates: hex_obj for hex_obj in element_dict.values()}
    ASPECT_RATIO = 1.7
    full_building_blocks = []
    size_x = int(round(size * ASPECT_RATIO)) + 2
    block_width = size_x + 2 * 1
    for ii in range(size):
        if ii == 0:
            new_line = ' ' * (1 - ii - 1) + "/" + overbar * (size_x + ii * 2) + "\\" + ' ' * (1 - ii - 1)
        else:
            new_line = ' ' * (1 - ii - 1) + "/" + ' ' * (size_x + ii * 2) + "\\" + ' ' * (1 - ii - 1)
        full_building_blocks.append(new_line)
    for ii in reversed(range(size)):
        if ii == 0:
            new_line = ' ' * (1 - ii - 1) + "\\" + '_' * (size_x + ii * 2) + "/" + ' ' * (1 - ii - 1)
        else:
            new_line = ' ' * (1 - ii - 1) + "\\" + ' ' * (size_x + ii * 2) + "/" + ' ' * (1 - ii - 1)
        full_building_blocks.append(new_line)
    # top_line_building_block = ' ' * (size)+ '_' * (size_x) + ' ' * (size)
    # empty_building_block = ' ' * block_width
    lines = []
    # for jj in range(width):
    #     lines[0] += empty_building_block if jj % 2 else top_line_building_block
    for ii in range(height * size * 2):
        new_line = ''
        for jj in range(width):
            # expected_rectangular_coordinates = (jj + min_x, (ii - size * (jj % 2) - ((min_x % 2)) + 1) // (size * 2) + min_y)
            expected_rectangular_coordinates = (jj + min_x, (ii - size * (jj % 2) - size * 2 * ((jj + 1) % 2)) // (size * 2) + min_y)
            offset_index = ii % (size*2)
            if jj == 0:
                new_line += ' ' * (size - 1 - min(offset_index, 2 * size - 1 - offset_index))
            building_block_index = (ii + size * (jj % 2)) % (size * 2)
            if expected_rectangular_coordinates in element_rectangular_dict.keys():
                # building_block = full_building_blocks[(ii + size * (jj % 2)) % (size*2)]
                # building_block = building_block[:5] + \
                #                      str(expected_rectangular_coordinates[0]) + ',' + str(expected_rectangular_coordinates[1]) \
                #                  + building_block[8:]
                # print(ii, jj, expected_rectangular_coordinates)
                new_line += full_building_blocks[building_block_index]
                # new_line += building_block
            else:
                new_line += ' ' * len(full_building_blocks[building_block_index])
        lines.append(new_line)
    for element in element_dict.values():
        jj, ii = element.rectangular_coordinates
        adjusted_rectangular_coordinates = (jj - min_x,
                                            ii + ((jj) % 2) - min_y)   # - (min_y % 2))
        x_center = int((size_x + size + 1) * (adjusted_rectangular_coordinates[0] + 0.5)) + 1
        y = int(size * 2 * adjusted_rectangular_coordinates[1] + 1 + (adjusted_rectangular_coordinates[0] % 2 + 1) * size - int((len(element.text) + 1) / 2))
        text_broken_up = element.text
        if len(text_broken_up) > 2 * size:
            text_broken_up = text_broken_up[:2 * size]
        vertical_offset = len(text_broken_up) // 2
        for ind, chars in enumerate(text_broken_up):
            if ind >= size * 2 - 1:
                break
            space = size_x + 2 * min(ind, abs(ind + 1 - size * 2))
            if space < len(chars):
                chars = chars[0:space]
            x = x_center - int(len(chars)/2)
            level = y + ind - vertical_offset
            lines[level] = lines[level][0: x] + chars + lines[level][x + len(chars):]
    lines = remove_empty_lines(lines)
    output = '\n'.join(lines)
    return output

# def remove_interhex_space(lines, size, width, block_width):
#     for ii in range(len(lines)):
#         for jj in range(width):
#             pass


def remove_empty_lines(lines):
    indices_with_material = []
    for ii, line in enumerate(lines):
        if np.any([True for char in line if char != ' ']):
            indices_with_material.append(ii)
    if len(indices_with_material) > 0:
        lines = lines[indices_with_material[0]: indices_with_material[-1] + 1]
    else:
        lines = []
    return lines

def one_hot(length, index):
    output = np.zeros(length)
    output[index] = 1
    return output

def generate_rhombal_array_indices(starting_coordinates, ending_coordinates):
    coordinate_list = [starting_coordinates]
    coordinate_list = list()
    axis_diffs = [ending_coordinate - starting_coordinate for
                  (starting_coordinate, ending_coordinate) in
                  zip(starting_coordinates, ending_coordinates)]
    pos_axis_diffs = np.abs(axis_diffs)
    sign_axis_diffs = np.sign(axis_diffs)
    if np.all(pos_axis_diffs):
        # the two hexes are not aligned along any axis
        long_axis = np.argmax(np.abs(axis_diffs))
        loop_axes = tuple({0, 1, 2} - {long_axis})
        for ii in range(int(pos_axis_diffs[loop_axes[0]] + 1)):
            for jj in range(int(pos_axis_diffs[loop_axes[1]] + 1)):
                coordinate_list.append(tuple(np.array(starting_coordinates)
                                             + ii * one_hot(3, loop_axes[0]) * sign_axis_diffs
                                             + jj * one_hot(3, loop_axes[1]) * sign_axis_diffs
                                             + (ii+jj) * one_hot(3, long_axis) * sign_axis_diffs
                                             ))
    else:
        # the two hexes are aligned so the result is a  'flat rhombus' (line)
        for ii in range(max(pos_axis_diffs) + 1):
            coordinate_list.append(tuple(np.array(starting_coordinates) + ii * sign_axis_diffs))
    return coordinate_list

def generate_radial_array_indices(radius, origin=None):
    if origin is None:
        origin = (0, 0, 0)
    coordinate_list = [origin]
    for xx in range(-radius, radius + 1):
        for yy in range(-radius, radius + 1):
            if xx == 0 and yy == 0:
                continue
            zz = - xx - yy
            if abs(zz) > radius:
                continue
            coordinate_list.append((xx + origin[0], yy + origin[1], zz + origin[2]))
    return coordinate_list


def generate_radial_hex_array(radius, default_obj=None, origin=None):
    """generates an empty array with given radius (radius = 0 gives a single hex)
    """
    # if type(default_obj) != Goes_In_Hex:
    #     default_obj = Goes_In_Hex(value=default_obj, text=[str(default_obj)])
    coordinate_list = generate_radial_array_indices(radius, origin)
    element_list = [Hex(coordinates, default_obj) for coordinates in coordinate_list]
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
    # G = Grid(element_list, size=3)
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
    G = Grid(element_list, size=7)
    g = generate_radial_hex_array(1)
    g = generate_radial_hex_array(3)
    R = g[-2:0, -1:0, 3:0]
    print(R)
    h = generate_radial_hex_array(3, False)
    h[(3, -2, -1)].change(True)
    print(h.where(True))
    print(h)
    for element in g.element_list:
        element.text = ['rad =', str(element.get_radius())]
    print(G)
    print(g)
    print(repr(g))
