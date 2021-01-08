from array_index_generation import generate_rhombal_array_indices
from math_utilities import complete_cubic_coordinates


def unpack_key(key, extents):
    if isinstance(key, int):
        key = (key, slice(None))
    if isinstance(key, tuple):
        if len(key) in [2, 3]:
            types = [type(ii) for ii in key]
            if any([False for ii in types if ii in (int, slice) or ii is None]):
                return None
            if len(key) - key.count(None) == 3:
                if types.count(int) == 3 or types.count(slice) == 3:
                    skip_index = 2
                elif types.count(int) in [1,2]:
                    skip_index = types.index(slice)
                key = tuple([key[ii] for ii in range(len(key)) if ii != skip_index])
            if isinstance(key[0], slice) or isinstance(key[1], slice) or (len(key) == 3 and isinstance(key[2], slice)):
                starting_index = list()
                ending_index = list()
                for key_index in range(len(key)):
                    if isinstance(key[key_index], slice):
                        starting_index_portion = key[key_index].start if key[key_index].start is not None else extents[key_index][0]
                        ending_index_portion = key[key_index].stop if key[key_index].stop is not None else extents[key_index][1]
                        starting_index.append(starting_index_portion)
                        ending_index.append(ending_index_portion)
                    else:
                        starting_index.append(key[key_index])
                        ending_index.append(key[key_index])
                starting_index = complete_cubic_coordinates(starting_index)
                ending_index = complete_cubic_coordinates(ending_index)
                coordinate_list = generate_rhombal_array_indices(starting_index, ending_index)
                return coordinate_list
            else:
                return complete_cubic_coordinates(key)
        else:
            return None
    elif isinstance(key, list):
        return complete_cubic_coordinates(key)
    else:
        return None
