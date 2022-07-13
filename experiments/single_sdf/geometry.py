'''
basic geometry to sdf
'''

from typing import Tuple, List
import numpy as np

Point = Tuple[float]

class Geometry:
    EPS = 1e-12

    def distance_from_segment_to_point(a, b, p):
        ans = min(np.linalg.norm(a - p), np.linalg.norm(b - p))
        if (np.linalg.norm(a - b) > Geometry.EPS
            and np.dot(p - a, b - a) > Geometry.EPS
                and np.dot(p - b, a - b) > Geometry.EPS):
            ans = abs(np.cross(p - a, b - a) / np.linalg.norm(b - a))
        return ans


class Shape:
    def sdf(self, p):
        pass


class Circle(Shape):

    def __init__(self, c: Point, r: float):
        self.c = c
        self.r = r

    def sdf(self, p: Point) -> float:
        return np.linalg.norm(p - self.c) - self.r


class Polygon(Shape):

    def __init__(self, v: List[Point]):
        self.v = v

    def sdf(self, p: Point) -> float:
        return -self.distance(p) if self.point_is_inside(p) else self.distance(p)

    def point_is_inside(self, p: Point) -> bool:
        angle_sum = 0
        L = len(self.v)
        for i in range(L):
            a = self.v[i]
            b = self.v[(i + 1) % L]
            angle_sum += np.arctan2(np.cross(a - p, b - p),
                                    np.dot(a - p, b - p))
        return abs(angle_sum) > 1

    def distance(self, p: Point) -> float:
        ans = Geometry.distance_from_segment_to_point(self.v[-1], self.v[0], p)
        for i in range(len(self.v) - 1):
            ans = min(ans, Geometry.distance_from_segment_to_point(
                self.v[i], self.v[i + 1], p))
        return ans
