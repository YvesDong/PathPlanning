"""
Environment for rrt_2D
@author: huiming zhou
"""


class Env:
    def __init__(self):
        self.x_range = (0, 10)
        self.y_range = (0, 10.5)
        self.obs_boundary = self.obs_boundary()
        self.obs_circle = self.obs_circle()
        self.obs_rectangle = self.obs_rectangle()

    @staticmethod
    def obs_boundary():
        obs_boundary = [
            [0, 0, 1, 10.5],
            [0, 10.5, 10, 1],
            [1, 0, 10, 1],
            [10, 1, 1, 10.5]
        ]
        return obs_boundary

    @staticmethod
    def obs_rectangle():
        obs_rectangle = [
            [6, 7, 1, 2.5], # [leftLowX,leftLowY, width, height]
            [6, 5, 2, 2],
            [2.6, 3, 4, 2],
            [3, 5, 1, 3]
        ]
        return obs_rectangle

    @staticmethod
    def obs_circle():
        obs_cir = [
            # [25, 8, 2],
            # [27, 9, 2],
            # [30, 11, 2],
            # [32, 13, 2],
            # [39, 16, 3],
            # [23, 9, 1],
            # [21, 10, 2],
            # [18, 13, 3],
            # [15, 17, 2],
            # [6, 5, 2]
            ]

        return obs_cir
