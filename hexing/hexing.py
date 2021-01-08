from copy import *

from array_index_generation import generate_rhombal_array_indices, generate_radial_array_indices, \
    generate_angular_rhombal_array_indices, generate_triangular_array_indices, generate_star_array_indices
from constants import defaul_sie
from math_utilities import complete_cubic_coordinates
from grid_utilities import unpack_key
from print_utilities import generate_visual_grid

# todo:
# remove element list,
# fix print text to consider new lines without requiring list of strings
# get rid of changing rectangular coordinates
# consider multiple versions of rectangular coordinates

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
        coordinates = complete_cubic_coordinates(coordinates)
        self.cubic_coordinates = coordinates
        self.rectangular_coordinates = (int(coordinates[0]), int(coordinates[2] + (coordinates[0] - (coordinates[0] % 2)) / 2))
        self.radius = None
        self.obj = obj
        self.text = str(obj).split('\n')

    def get_radius(self):
        if self.radius is None:
            self.radius = max([abs(c) for c in self.cubic_coordinates])
        return self.radius

    def __deepcopy__(self, memodict={}):
        new = Hex(self.cubic_coordinates, deepcopy(self.obj))
        return new

    def __copy__(self, memodict={}):
        new = Hex(self.cubic_coordinates, copy(self.obj))
        return new

    def __eq__(self, other):
        return self.obj == other.obj

    def update_rectangular_coordinates(self, origin):
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


class Grid:
    """heagonal grid consisting of a hexagonal elements"""

    def empty(self):
        self.element_dict = {}
        self.origin = None
        self.min_x = None
        self.min_y = None
        self.max_x = None
        self.max_y = None
        self.extents = None

    def __init__(self, elements=None, size=defaul_sie):
        self.size=size
        if elements is None:
            self.empty()
            return
        if isinstance(elements, Hex):
            elements = [elements]
        try:
            if len(elements) == 0:
                self.empty()
                return
        except TypeError:
            self.empty()
            return

        if isinstance(elements, dict):
            self.element_dict = {coordinates: element if isinstance(element, Hex) else Hex(coordinates, element) for coordinates, element in elements.items()}
            # if isinstance(elements.values()[0], Hex):
            #     self.element_dict = {coordinates: element for coordinates, element in elements.items()}
        elif isinstance(elements, list):
            if isinstance(elements[0], Hex):
                self.element_dict = {element.cubic_coordinates: element for element in elements}
            elif (isinstance(elements[0], list) or isinstance(elements[0], tuple)) and len(elements[0]) == 2:
                self.element_dict = {coordinates: Hex(coordinates, element) for coordinates, element in elements}
            else:
                raise TypeError(elements)
        self.element_list = list(self.element_dict.values())
        self.origin = None
        rectangular_coordinates_list = [element.rectangular_coordinates for element in self.element_dict.values()]
        x_list = [coord[0] for coord in rectangular_coordinates_list]
        y_list = [coord[1] for coord in rectangular_coordinates_list]
        self.min_x = int(min(x_list))
        self.min_y = int(min(y_list))
        self.max_x = int(max(x_list))
        self.max_y = int(max(y_list))
        # self.element_dict = {element.cubic_coordinates: element for element in self.element_list}
        xx_list = [coordinate[0] for coordinate in list(self.element_dict.keys())]
        yy_list = [coordinate[1] for coordinate in list(self.element_dict.keys())]
        zz_list = [coordinate[2] for coordinate in list(self.element_dict.keys())]
        self.extents = [[min(xx_list), max(xx_list)],
                        [min(yy_list), max(yy_list)],
                        [min(zz_list), max(zz_list)]]

    def __deepcopy__(self, memodict={}):
        new = Grid([deepcopy(element) for element in self.element_list], size=self.size)
        # new.element_list = [deepcopy(element) for element in self.element_list]
        # new.origin = self.origin
        # new.min_x = self.min_x
        # new.min_y = self.min_y
        # new.max_x = self.max_x
        # new.max_y = self.max_y
        # new.element_dict = {element.cubic_coordinates: element for element in new.element_list}
        return new

    def __copy__(self):
        new = Grid([copy(element) for element in self.element_list], size=self.size)
        return new

    def __eq__(self, other):
        if len(self.element_list) != len(other.element_list):
            return False
        for coordinates in self.element_dict:
            if coordinates not in other.element_dict or self.element_dict[coordinates] != other.element_dict[coordinates]:
                return False
        return True

    def get_index_list(self):
        return set(self.element_dict.keys())

    def unary_elementwise_operation(self, operation, arg=None):
        new = copy(self)
        for element in new.element_list:
            operation_funx = getattr(element.obj, operation)
            if arg is None:
                element.change(operation_funx())
            else:
                element.change(operation_funx(arg))
        return new

    def binary_elementwise_operation(self, other, operation):
        if isinstance(other, Grid) and self.get_index_list() != other.get_index_list():
            raise IndexError('cannot perform element-wise operation with Grids of different shapes')
        new = copy(self)
        for element in new.element_list:
            operation_funx = getattr(element.obj, operation)
            other_obj = other.element_dict[element.cubic_coordinates].obj if isinstance(other, Grid) else other
            element.change(operation_funx(other_obj))
        return new

    def elementwise_augmentation_operation(self, other, operation):
        if isinstance(other, Grid) and self.get_index_list() != other.get_index_list():
            raise IndexError('cannot perform element-wise operation with Grids of different shapes')
        base_operation = operation.replace('__i', '__')
        for element in self.element_list:
            operation_funx = getattr(element.obj, base_operation)
            other_obj = other.element_dict[element.cubic_coordinates].obj if isinstance(other, Grid) else other
            element.change(operation_funx(other_obj))
        return element.obj

    def __neg__(self):
        return self.unary_elementwise_operation('__neg__')

    def __pos__(self):
        return self.unary_elementwise_operation('__pos__')

    def __abs__(self):
        return self.unary_elementwise_operation('__abs__')

    def __round__(self, n=0):
        return self.unary_elementwise_operation('__round__', arg=n)

    def __floor__(self):
        return self.unary_elementwise_operation('__floor__')

    def __ceil__(self):
        return self.unary_elementwise_operation('__ceil__')

    def __trunc__(self):
        return self.unary_elementwise_operation('__trunc__')

    def __add__(self, other):
        return self.binary_elementwise_operation(other, '__add__')

    def __or__(self, other):
        return self.binary_elementwise_operation(other, '__or__')

    def __and__(self, other):
        return self.binary_elementwise_operation(other, '__and__')

    def __xor__(self, other):
        return self.binary_elementwise_operation(other, '__xor__')

    def __lshift__(self, other):
        return self.binary_elementwise_operation(other, '__lshift__')

    def __rshift__(self, other):
        return self.binary_elementwise_operation(other, '__rshift__')

    def __sub__(self, other):
        return self.binary_elementwise_operation(other, '__sub__')

    def __mul__(self, other):
        return self.binary_elementwise_operation(other, '__mul__')

    def __floordiv__(self, other):
        return self.binary_elementwise_operation(other, '__floordiv__')

    def __truediv__(self, other):
        return self.binary_elementwise_operation(other, '__truediv__')

    def __mod__(self, other):
        return self.binary_elementwise_operation(other, '__mod__')

    def __pow__(self, other):
        return self.binary_elementwise_operation(other, '__pow__')

    def __lt__(self, other):
        return self.binary_elementwise_operation(other, '__lt__')

    def __le__(self, other):
        return self.binary_elementwise_operation(other, '__le__')

    def __ge__(self, other):
        return self.binary_elementwise_operation(other, '__ge__')

    def __eq__(self, other):
        return self.binary_elementwise_operation(other, '__eq__')

    def __ne__(self, other):
        return self.binary_elementwise_operation(other, '__ne__')

    def __iadd__(self, other):
        return self.elementwise_augmentation_operation(other, '__iadd__')

    def __isub__(self, other):
        return self.elementwise_augmentation_operation(other, '__isub__')

    def __imul__(self, other):
        return self.elementwise_augmentation_operation(other, '__imul__')

    def __ifloordiv__(self, other):
        return self.elementwise_augmentation_operation(other, '__ifloordiv__')

    def __itruediv__(self, other):
        return self.elementwise_augmentation_operation(other, '__itruediv__')

    def __imod__(self, other):
        return self.elementwise_augmentation_operation(other, '__imod__')

    def __ipow__(self, other):
        return self.elementwise_augmentation_operation(other, '__ipow__')

    def __ilshift__(self, other):
        return self.elementwise_augmentation_operation(other, '__ilshift__')

    def __irshift__(self, other):
        return self.elementwise_augmentation_operation(other, '__irshift__')

    def __iand__(self, other):
        return self.elementwise_augmentation_operation(other, '__iand__')

    def __ior__(self, other):
        return self.elementwise_augmentation_operation(other, '__ior__')

    def __ixor__(self, other):
        return self.elementwise_augmentation_operation(other, '__ixor__')


    def update_min_max(self, cubic_coordinates, rectangular_coordinates):

        if self.min_x is None or rectangular_coordinates[0] < self.min_x:
            self.min_x = int(rectangular_coordinates[0])
        if self.max_x is None or rectangular_coordinates[0] > self.max_x:
            self.max_x = int(rectangular_coordinates[0])
        if self.min_y is None or rectangular_coordinates[1] < self.min_y:
            self.min_y = int(rectangular_coordinates[1])
        if self.max_y is None or rectangular_coordinates[1] > self.max_y:
            self.max_y = int(rectangular_coordinates[1])
        if self.extents is None:
            self.extents = [[cubic_coordinates[0], cubic_coordinates[0]],
                            [cubic_coordinates[1], cubic_coordinates[1]],
                            [cubic_coordinates[2], cubic_coordinates[2]],]
        else:
            for index in range(len(cubic_coordinates)):
                if self.extents[index][0] > cubic_coordinates[index]:
                    self.extents[index][0] = cubic_coordinates[index]
                if self.extents[index][1] < cubic_coordinates[index]:
                    self.extents[index][1] = cubic_coordinates[index]

    def add_hex(self, hex):
        if hex.cubic_coordinates in self.element_dict:
            raise NameError('location already taken')
        # hex.update_rectangular_coordinates(self.origin) #fix?
        self.element_list.append(hex)
        self.element_dict[hex.cubic_coordinates] = hex
        self.update_min_max(hex.cubic_coordinates, hex.rectangular_coordinates)

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
            return Grid([self.element_dict[coordinates] for coordinates in coordinate_list if coordinates in self.element_dict], size=self.size)
        except:
            raise IndexError

    def __getitem__(self, key):
        unpacked_key = unpack_key(key, self.extents)
        if isinstance(unpacked_key, list):
            return self.get_subgrid(unpacked_key)
        elif isinstance(unpacked_key, tuple):
            try:
                return self.element_dict[unpacked_key].obj
            except KeyError as er:
                raise IndexError(er)
        else:
            return None

    def __setitem__(self, key, value):
        if isinstance(value, Grid):
            pass #todo set to a slice if shape/some dimensions match?
        else:
            unpacked_key = unpack_key(key, self.extents)
            if isinstance(unpacked_key, list):
                for coordinate in unpacked_key:
                    if coordinate in self.element_dict:
                        self.element_dict[coordinate].change(value)
                    else:
                        self.add_hex(Hex(coordinate, value))
            elif isinstance(unpacked_key, tuple):
                if unpacked_key in self.element_dict:
                    self.element_dict[unpacked_key].change(value)
                else:
                    self.add_hex(Hex(unpacked_key, value))
            else:
                pass

    def __delitem__(self, key):
        unpacked_key = unpack_key(key, self.extents)
        if isinstance(unpacked_key, list):
            for coordinate in unpacked_key:
                if coordinate in self.element_dict:
                    self.element_dict.pop(coordinate)
                else:
                    pass
        elif isinstance(unpacked_key, tuple):
            if unpacked_key in self.element_dict:
                self.element_dict.pop(unpacked_key)
                #remove from element_list
            else:
                pass
        else:
            pass

    def __str__(self):
        return generate_visual_grid(self.element_dict, self.min_x, self.min_y, self.max_x, self.max_y, size=self.size)

    def __repr__(self):
        return f'Grid({self.element_list})'


def full(mode='radial', fill_value=None, origin=None, radius=None, starting_coordinates=None, ending_coordinates=None,
         side_length=None, orientation=None, size=defaul_sie):

    if mode == 'radial':
        assert isinstance(radius, int)
        coordinate_list = generate_radial_array_indices(radius, origin)
    elif mode == 'star':
        assert isinstance(radius, int)
        coordinate_list = generate_star_array_indices(radius, origin)
    elif mode == 'rhombal':
        assert isinstance(starting_coordinates, int) and isinstance(ending_coordinates, int)
        coordinate_list = generate_rhombal_array_indices(starting_coordinates, ending_coordinates)
    elif mode == 'triangular':
        assert isinstance(side_length, int)
        coordinate_list = generate_triangular_array_indices(side_length, orientation, origin)
    elif mode == 'angular_rhomal':
        assert isinstance(side_length, int)
        coordinate_list = generate_angular_rhombal_array_indices(side_length, orientation, origin)
    else:
        coordinate_list = [(0, 0, 0)]

    element_list = [Hex(coordinates, fill_value) for coordinates in coordinate_list]
    grid = Grid(element_list, size=size)
    return grid


def generate_radial_hex_array(radius, fill_value=None, origin=None):
    """generates an empty array with given radius (radius = 0 gives a single hex)
    """
    # if type(fill_value) != Goes_In_Hex:
    #     fill_value = Goes_In_Hex(value=fill_value, text=[str(fill_value)])
    coordinate_list = generate_radial_array_indices(radius, origin)
    element_list = [Hex(coordinates, fill_value) for coordinates in coordinate_list]
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
    t = full(mode='triangular', side_length=3, orientation=3, fill_value='tri')
    print(t)
    r = full(mode='radial', radius=2, fill_value='rad')
    print(r)
    s = full(mode='star', radius=3, fill_value='star')
    print(s)
    s = full(mode='star', radius=4, fill_value='s', size=2)
    print(s)



    h = generate_radial_hex_array(3, False)
    # h[(3, -2, -1)].change(True)
    h[3, -1, -2] = True
    print(G[:, :])
    # G[0, :, 0]
    G[0, 0] = 1
    G[0, :, :]
    print(G[:, 0])
    print(G[0:0, 0])
    print(G[None, 0, :])
    h[-2:0, -1:0, 3:0] = 5
    print(h.where(True))
    print(h.where(5))
    print(h)
    h[h.where(5)] += 6
    print(h)
    for element in g.element_list:
        element.text = ['rad =', str(element.get_radius())]
    print(G)
    print(g)
    print(repr(g))
    num = generate_radial_hex_array(2, 3)
    print(num + num)
    -num
