from numpy import (arange, array, complex64, complex128, float32, float64,
                   fromfunction, ones, tensordot)
from numpy.linalg import LinAlgError, tensorinv
from numpy.random import RandomState
from numpy.testing import (assert_allclose, assert_raises_regex,
                           run_module_suite)

# TODO: Only test if tdot(tinv(t), t) == id in all cases?
#       (Start with more extensive tensordot() tests?)

# TODO: Some close to numerically ill-conditioned cases?

# What types shall we test and what accuracy to expect.
# TODO: With the 64-bit floats rtol=1e-11 should work??
tolerances = {
    'r': ((float32, 1e-4), (float64, 1e-4)),
    'c': ((complex64, 1e-4), (complex128, 1e-4))
}


def identity(shape, dtype=float32):
    """
    Returns the identity in the group of I1x...xInxI1x...xIn tensors with *n.
    """
    hndim, odd = divmod(len(shape), 2)
    if odd or shape[:hndim] != shape[hndim:]:
        raise ValueError("Shape has to match (I1, ..., In, I1, ..., In)")
    eye = ones(shape=shape, dtype=bool)
    for axis in range(hndim):
        eye &= fromfunction(lambda *i: i[axis] == i[hndim + axis], shape)
    return eye.astype(dtype)


class TestTensorinv:
    def identity(self, tensor, fields, ind):
        for field in fields:
            # TODO: This would better use rtol not atol.
            for dtype, atol in tolerances[field]:
                tensor = array(tensor, dtype=dtype)
                inverse = tensorinv(tensor, ind=ind)
                product = tensordot(inverse, tensor, axes=ind)
                expected = identity(2 * tensor.shape[ind:])
                assert_allclose(product, expected, atol=atol)

    def inverse(self, tensor, inverse, fields, ind):
        for field in fields:
            # TODO: This would better use rtol not atol.
            for dtype, atol in tolerances[field]:
                tensor = array(tensor, dtype=dtype)
                inverse = array(inverse, dtype=dtype)
                assert_allclose(tensorinv(tensor, ind=ind), inverse, atol=atol)

    def singular(self, tensor, fields, ind):
        for field in fields:
            for dtype, _ in tolerances[field]:
                tensor = array(tensor, dtype=dtype)
                with assert_raises_regex(LinAlgError, '[Ss]ingular'):
                    # TODO: Singular with order > 2 gives "Singular matrix".
                    tensorinv(tensor, ind=ind)

    def nonsquare(self, tensor, ind):
        tensor = array(tensor)
        with assert_raises_regex(LinAlgError, 'square'):
            # TODO: "Last 2 dimensions (...) must be square" -- not quite.
            tensorinv(tensor, ind=ind)

    def test_scalar(self):
        self.inverse([1], [1], 'rc', 1)  # TODO: ind = 0?
        self.inverse([-1], [-1], 'rc', 1)
        self.inverse([2], [.5], 'rc', 1)
        self.inverse([1+2j], [.2-.4j], 'c', 1)

    def test_matrix(self):
        t = [[1, 0], [0, 1]]
        self.inverse(t, t, 'rc', 1)
        t = [[-2, 1], [-3, 2]]
        self.inverse(t, t, 'rc', 1)
        t = [[0, 1], [2, 3]]
        i = [[-3/2, 1/2], [1, 0]]
        self.inverse(t, i, 'rc', 1)
        self.inverse(i, t, 'rc', 1)
        t = [[0+1j, 2+3j], [4+5j, 6+7j]]
        i = [[-.4375+.375j, .1875-.125j], [.3125-.25j, -.0625+0.j]]
        self.inverse(t, i, 'c', 1)
        self.inverse(i, t, 'c', 1)

    def test_triad(self):
        # If T is 4x2x2 then inv(T, ind=1) gives T^-1 of shape 2x2x4, such
        # that T^-1 *1 T is the 2x2x2x2 tetrad identity (with respect to *2)
        # and T *2 T^-1 is the 2x2 identity matrix (with respect to *1).
        t = [[[0, 1], [2, 3]], [[3, 0], [1, 2]],
             [[2, 3], [0, 1]], [[1, 2], [3, 0]]]
        i = [[[-5/24, 7/24, 1/24, 1/24], [1/24, -5/24, 7/24, 1/24]],
             [[1/24, 1/24, -5/24, 7/24], [7/24, 1/24, 1/24, -5/24]]]
        self.inverse(t, i, 'rc', 1)
        self.inverse(i, t, 'rc', 2)
        t = [[[0, 1], [1, 1j]], [[0, 1], [-1, 2j]],
             [[3j, -1], [1, 0]], [[4j, -1], [-1, 0]]]
        i = [[[-.4j, .2j, .2j, -.4j], [1.4, -.7, -1.2, .9]],
             [[.2, -.1, .4, -.3], [.6j, -.8j, -.8j, .6j]]]
        self.inverse(t, i, 'c', 1)
        self.inverse(i, t, 'c', 2)

    def test_tetrad(self):
        t = [[[[1, 0], [0, 0]], [[0, 1], [0, 0]]],
             [[[0, 0], [1, 0]], [[0, 0], [0, 1]]]]
        self.inverse(t, t, 'rc', 2)
        t = [[[[.5, -.5], [-.5, -.5]], [[-.5, .5], [-.5, -.5]]],
             [[[-.5, -.5], [.5, -.5]], [[-.5, -.5], [-.5, .5]]]]
        self.inverse(t, t, 'rc', 2)
        t = [[[[0, 1/3], [2/3, 1]], [[1, 0], [1/3, 2/3]]],
             [[[2/3, 1], [0, 1/3]], [[1/3, 2/3], [1, 0]]]]
        i = [[[[-5/8, 7/8], [1/8, 1/8]], [[1/8, -5/8], [7/8, 1/8]]],
             [[[1/8, 1/8], [-5/8, 7/8]], [[7/8, 1/8], [1/8, -5/8]]]]
        self.inverse(t, i, 'rc', 2)
        self.inverse(i, t, 'rc', 2)
        t = [[[[1, 0], [0, 2j]], [[0, 3], [4j, 0]]],
             [[[5, 0], [0, 6j]], [[0, 7], [8j, 0]]]]
        i = [[[[-1.5, 0], [.5, 0]], [[0, -2], [0, 1]]],
             [[[0, -1.75j], [0, .75j]], [[-1.25j, 0], [.25j, 0]]]]
        self.inverse(t, i, 'c', 2)
        self.inverse(i, t, 'c', 2)

        # Inverse of 2x3x3x2 should be 3x2x2x3 (giving 2x3x2x3 or 3x2x3x2
        # identities with respect to *2).
        t = [[[[0, 1j], [2, -2], [-1, 0]],
              [[1, -1], [0, 0], [-1, 1]],
              [[1, 0], [-2, 2], [0, -1]]],
             [[[0, 1j], [2, -4], [-1, 0]],
              [[1, -2], [0, 0], [-2, 1]],
              [[1, 0], [-4, 2], [0, -2]]]]
        i = [[[[-1, 0, 2], [1, 0, -1]],
              [[1.5-1.5j, 1.5-1.5j, -1.5+1.5j], [-1+1j, -1+1j, 1-1j]]],
             [[[-.5, -1, 1.5], [.5, .5, -1]],
              [[.5, 0, 0], [-.5, 0, 0]]],
             [[[-1.5+1.5j, -.5+1.5j, 1.5-1.5j], [1-1j, -1j, -1+1j]],
              [[1, 2, -2], [-1, -1, 1]]]]
        self.inverse(t, i, 'c', 2)
        self.inverse(i, t, 'c', 2)

    def test_high_order(self):
        r = RandomState(26)
        t = r.uniform(size=(3, 3, 3, 3, 3, 3))
        self.identity(t, 'rc', 3)
        t = r.uniform(size=(256, 2, 2, 2, 2, 4, 4))
        self.identity(t, 'rc', 1)
        t = r.uniform(size=(1, 2, 1, 3, 1, 4, 1, 3, 1, 8))
        self.identity(t, 'rc', 6)
        t = 1 + 1j * r.uniform(size=(2, 3, 4, 4, 6))
        self.identity(t, 'c', 3)

    def test_singular(self):
        self.singular([0], 'rc', 1)
        self.singular([[1, 0], [0, 0]], 'rc', 1)
        self.singular([[1+0j, 1+2j], [1+0j, 1+2j]], 'c', 1)
        self.singular([[[0, 1], [1, 1j]], [[0, 2], [2, 1j]],
                       [[0, 3], [3, 1j]], [[0, 4], [4, 1j]]], 'c', 1)
        self.singular([[[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
                       [[[8, 9], [10, 11]], [[12, 13], [14, 15]]]], 'rc', 2)
        # TODO: Real triad, complex tetrad, some higher order.

    def test_nonsquare(self):
        self.nonsquare(ones(3), 1)
        self.nonsquare(ones((2, 3)), 1)
        self.nonsquare(ones((4, 4, 4, 4)), 1)


if __name__ == '__main__':
    run_module_suite()
