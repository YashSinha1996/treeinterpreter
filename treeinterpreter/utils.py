from collections import Counter


def aggregated_contribution(contribution_map):
    contr_sum  = {}
    for j, dct in enumerate(contribution_map):
        for k in set(dct.keys()).union(set(contr_sum.keys())):
            contr_sum[k] = (contr_sum.get(k, 0)*j + dct.get(k,0) ) / (j+1)
    return contr_sum


def compare_zero(count):
    if hasattr(count, "__iter__"):
        return any(count)
    return count > 0


class MultiCount(Counter):

    def __add__(self, other):

        if not isinstance(other, Counter):
            return NotImplemented
        result = MultiCount()
        for elem, count in self.items():
            newcount = count + other[elem]
            if compare_zero(newcount):
                result[elem] = newcount
        for elem, count in other.items():
            if elem not in self and compare_zero(count):
                result[elem] = count
        return result

    def __sub__(self, other):

        if not isinstance(other, Counter):
            return NotImplemented
        result = MultiCount()
        for elem, count in self.items():
            newcount = count - other[elem]
            if compare_zero(newcount):
                result[elem] = newcount
        for elem, count in other.items():
            if elem not in self and compare_zero(count):
                result[elem] = 0 - count
        return result

    def __repr__(self):
        return str(dict(self.items()))

    def __mul__(self, other):
        result = MultiCount()
        for elem, count in self.items():
            newcount = count * other
            if compare_zero(newcount):
                result[elem] = newcount
        return result


if __name__ == '__main__':
    import numpy as np
    a = MultiCount()
    b = MultiCount()
    a[(1, 2)] = np.array([2, 3])
    b[(1, 2)] = np.array([3, 4])
    b[(2, 3)] = np.array([0, 2])

    print(a)
    print(b)
    print(a + b)
