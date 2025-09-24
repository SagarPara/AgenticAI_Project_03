import pytest
from square import get_square

def test_sq():
    x = 5
    res = get_square(5)
    assert res == 25 # assert used for test
    assert get_square(3) == 9
