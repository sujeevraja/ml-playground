#!/usr/bin/env python3

"""
Run this script for examples on policy iteration.
"""

import logging
import numpy as np
import typing

log = logging.getLogger(__name__)


class IteratePolicy:
    """Class to find the value function objected by iterating a given policy
    over a square grid with given size.

    Grid squares are numbered horizontally from the top row starting from 0.
    The first and last squares are considered terminal states. The policy
    iteration is performed by directly using the Bellman equation without
    depending on libraries. This for me to fully understand how the equation
    works.
    """

    def __init__(self, square_length: int):
        # Actions are up/down/left/right directions on a 4X4 grid.
        self.actions = ['N', 'S', 'E', 'W']

        # Reward is -1 for any move and denotes consumption of a time step.
        self._reward = -1

        # Number of squares in 1 side of the grid.
        self.size = square_length

        self.final_state = (square_length * square_length) - 1
        self.terminal_states = [0, self.final_state]

    def run(self, policy: typing.List[float], iterations: int):
        """Solve the Bellman iterative equation with the given policy."""
        value = np.zeros((self.size, self.size))
        for i in range(iterations):
            log.info(f"\n{value}")

    def neighbors(self, i: int) -> typing.List[int]:
        """Return valid neighbors obtained by taking 1 step up/down/left/right
        from the grid square with the given index `i`."""
        # Don't allow stepping away from a terminal state.
        if i == 0 or i == self.final_state:
            return [i] * self.size

        # Assume that if a neighbor index falls outside the grid, then an
        # action stepping to that # neighbor should be treated as staying in
        # the same grid.
        neighbors = []
        for n in [i - 1, i + 1, i + self.size, i - self.size]:
            neighbors.append(n if 0 <= n <= self.final_state else i)
        return neighbors


def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s--: %(message)s',
                        level=logging.INFO)
    IteratePolicy(square_length=4).run([0.25, 0.25, 0.25, 0.25], 2)


if __name__ == '__main__':
    main()
