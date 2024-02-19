import json
from enum import IntEnum
from random import random, seed
from random import uniform as u

import matplotlib.pyplot as plt
import numpy as np
import shapely
from shapely.geometry import Polygon


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class Space:
    def __init__(self, division: int):
        self.polygons = []
        self.overlapping = 0.0
        self.outliers = 0.0
        self.main = Polygon(
            [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]
        )
        self.division_table = np.array([1 / (2**i) for i in range(division)])
        self.division_table = softmax(self.division_table)

    def place(self, polygon: Polygon, turn: float, pos: np.array):
        polygon = shapely.affinity.rotate(polygon, turn * 2 * np.pi)
        pos = np.sum(pos * self.division_table, axis=1)
        print(pos)
        polygon = shapely.affinity.translate(polygon, pos[0], pos[1])

        for pol2 in self.polygons:
            if shapely.intersects(polygon, pol2):
                area = polygon.intersection(pol2).area
                self.overlapping += area

        diff_area = polygon.difference(self.main).area
        self.outliers += diff_area
        self.polygons.append(polygon)


class RealShapeGenerator:
    def __init__(
        self,
        scale: float = 1,
        fixed_seed: int = 1701,
    ):
        super().__init__()
        self.scale_factor = scale
        self.shape_buffer = self.load_fixed_shapes()
        if fixed_seed is not None:
            seed(fixed_seed)

    def get_next_n_polygons(self, n):
        return [self.get_polygon() for _ in range(n)]

    def load_fixed_shapes(self):
        with open("fixed_shapes.json", "r") as f:
            fixed_shapes = json.load(f)

        return fixed_shapes

    def get_polygon(self):
        if len(self.shape_buffer) == 0:
            copied_buffer = self.load_fixed_shapes()
            polygonized_buffer = [Polygon(item) for item in copied_buffer]
            scaled_buffer = [
                shapely.affinity.scale(
                    item, xfact=(random() / 2) + (5 / 6), yfact=(random() / 2) + (5 / 6)
                )
                for item in polygonized_buffer
            ]
            rotated_buffer = [
                shapely.affinity.rotate(item, random() * 90) for item in scaled_buffer
            ]
            coords_buffer = [list(item.exterior.coords) for item in rotated_buffer]
            self.shape_buffer = coords_buffer
        s = self.shape_buffer.pop()
        poly = Polygon(s)
        poly = shapely.affinity.scale(
            poly, xfact=self.scale_factor, yfact=self.scale_factor
        )
        poly = shapely.affinity.translate(
            poly, -poly.exterior.coords[0][0], -poly.exterior.coords[0][1]
        )
        return poly
