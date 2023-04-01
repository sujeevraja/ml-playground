#!/usr/bin/env python3

"""
Run this script for examples on policy iteration.
"""

import logging
import numpy as np
import typing

log = logging.getLogger(__name__)


class PolicyIteration:
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

        # Number of squares in 1 side of the grid.
        self.size = square_length
        self.num_squares = self.size * self.size

        self.final_state = (square_length * square_length) - 1
        self.terminal_states = [0, self.final_state]

        # Assume that there is no need for a discount factor as all paths
        # are likely to terminate in a terminal state.
        self.discount_factor = 1.0

    def run(self, policy: typing.Dict[str, float], iterations: int):
        """Solve the Bellman iterative equation with the given policy."""
        # Note that `rewards` below is an expected reward computed as
        # R'_s = Expectation[R_{t+1} | S_t = s]. However, as there is no
        # discount factor and all rewards for all transitions are just -1,
        # we can say that the expected reward is -1 for all transitions out
        # of all non-terminal states and 0 for transitions out of terminal
        # states.
        rewards = np.full((self.num_squares), -1.0)
        rewards[0] = rewards[-1] = 0
        log.info(f"expected rewards: {rewards}")

        stm = self.build_state_transition_matrix(policy)
        log.info(f"state transition matrix:\n{stm}")
        value = np.zeros((self.num_squares))
        log.info(f"initial value:\n{value}")
        for i in range(iterations):
            value = np.transpose(rewards) + \
                np.matmul(stm, np.transpose(value))

            if (i+1) in {1, 2, 3, 10} or i == iterations - 1:
                log.info(f"iteration {i+1}:\n{self._prettify(value)}")

    def _prettify(self, value):
        return np.round(value.reshape(self.size, self.size), 1)

    def build_state_transition_matrix(self, policy: typing.Dict[str, float]):
        """Build state transition matrix for the given policy.

        To clarify what this is, we set up the following notation:
        > A is the set of actions with cardinality |A|.
        > S is the set of states with cardinality |S|.
        > \pi(a|s) = Prob[action = a | state = s] for a \in A, s \in S.
        > P^a is the state transition probability matrix for fixed action a.
          It has dimension (|S|x|S|). P^a(s1,s2) is the probability of
          transitioning from state s1 to state s2 (s1 being the row index and
          s2 being the column index). In our case, for fixed a, P^a(s1,s2) is
          1 if s2 is the neighbor of s1 for action a and 0 otherwise.

        As the given policy depends only on the action and is independent of
        the state s, \pi(a|s) is just a scalar value for fixed a. To clarify
        this independence of state, we can just denote the policy as \pi(a).

        We want to compute and return P = \sum_{a \in A} \pi(a) P^a.
        """
        mat = np.zeros((self.num_squares, self.num_squares))
        for i in range(self.num_squares):
            for a in self.actions:
                n = self.get_neighbor(i, a)
                mat[i][n] += policy[a]

        return mat

    def get_neighbor(self, i: int, a: str) -> int:
        """Find neighbor of square with given number for given action.

        Assume that:
        - If the square is terminal, then its neighbor is always itself.
        - If the action takes us outside the grid, the neighbor of the square
            for the corresponding action is itself.

        Args:
            i: square index (starting from 0)
            a: action (one of 'N', 'S', 'E', 'W')

        Returns:
            int: neighbor square index (may be the square itself)

        """
        if i in self.terminal_states:
            return i

        if a == 'N':
            n = i - self.size
            return n if n >= 0 else i

        if a == 'S':
            n = i + self.size
            return n if n <= self.final_state else i

        if a == 'E':
            n = i + 1
            return n if (i % self.size) != (self.size - 1) else i

        if a == 'W':
            n = i - 1
            return n if ((i % self.size) != 0) else i


def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s--: %(message)s',
                        level=logging.INFO)

    # avoid line wrapping when printing numpy objects
    np.set_printoptions(linewidth=np.inf)

    policy = {
        'N': 0.25,
        'S': 0.25,
        'E': 0.25,
        'W': 0.25,
    }
    num_squares_per_side = 4
    PolicyIteration(num_squares_per_side).run(policy, 115)


if __name__ == '__main__':
    main()
