import openpyscad as ops
from functools import reduce

class Representation:
    def __init__(self, points):
        self.points = points
        self.time_steps = len(self.points)
        self.rects = self.transform_points()
        self.shape = self.build_shape()
        self.total = self.sum()

    def transform_points(self):
        transform = lambda x: [1] + x
        return list(map(transform, self.points))

    def build_shape(self):
        shape = []
        for t in range(self.time_steps):
            shape.append(ops.Cube(self.rects[t]).translate([t,0,0]))
        return shape

    def sum(self):
        union = lambda r1, r2: r1 + r2
        return reduce(union, self.shape)

    def get_rects(self):
        return self.rects

    def get_final_shape(self):
        return self.total
