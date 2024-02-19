import numpy as np

from plotting import show_in_pyplot
from polygons import RealShapeGenerator, Space

g = RealShapeGenerator(scale=0.0025)
s = Space(3)
pols = g.get_first_n_polygons(20)
for pol in pols:
    turn = np.ones(1)
    pos = np.random.rand(2, 3)
    s.place(pol, turn, pos)

print(s.outliers, s.overlapping)

show_in_pyplot(s.polygons, s.main)
