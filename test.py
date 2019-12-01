
import numpy as np
from dwt import harr_decomposition, harr_recomposition, harr2d_decomp
import pywt
from numpy.testing import assert_allclose, assert_, assert_raises


def test_harr_1D_transform():
    level = 3
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    assert isinstance(level, int), "provided {} is not int".format(level)

    if level == 1:
        cA_expect, cD_expect = pywt.dwt(x, 'db1')
        cA, cD = harr_decomposition(x)

        print(cA, cA_expect)
        print(cD, cD_expect)

        assert_allclose(cA, cA_expect, rtol=1e-5)
        # assert_allclose(cD, cD_expect, rtol=1e-5)

        x_rec = harr_recomposition(cA, cD)
        print(x_rec)
        assert_allclose(x_rec, x, rtol=1e-5)

    if level > 1:
        coeffs_expect = pywt.wavedec(x, 'db1', level)
        print(coeffs_expect)
        coeffs = harr_decomposition(x, level=level)
        print('\n')
        print(coeffs)
        # assert_allclose(coeffs, coeffs_expect, rtol=1e-5)

        x_rec = harr_recomposition(coeffs)
        print(x_rec)
        assert_allclose(x_rec, x, rtol=1e-5)





if __name__ == "__main__":
    test_harr_1D_transform()
