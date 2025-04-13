#!/usr/bin/env python3

"""
Run this script for an example on evaluating the state value function for a
given policy on a nxn grid world with the top-left and bottom-right squares as
terminal states. This script solves example 4.1 from the following book:

Sutton, Richard S., and Andrew G. Barto.
Reinforcement learning: An introduction.
MIT press, 2018.

Verify this script by running `test_evaluate_policy.py`.
"""

import logging
import numpy as np
import typing

log = logging.getLogger(__name__)


class GridWorld(typing.NamedTuple):
    """A nxn grid with top-left and bottom-right suares as terminal states.

    The squares are numbered from left to right and row by row with the
    top-left square numbered as 0 and the bottom-right square numbered as
    (n*n - 1).

    Attributes:
        size: number of squares on 1 side of the grid.
        last_index: index of bottom-right square.
    """
    size: int = 4
    num_squares: int = size * size
    last_index: int = num_squares - 1
    terminal: typing.Set[int] = {0, last_index}

    def neighbor(self, i: int, a: str) -> int:
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
        if i in self.terminal:
            return i

        if a == 'N':
            n = i - self.size
            return n if n >= 0 else i

        if a == 'S':
            n = i + self.size
            return n if n <= self.last_index else i

        if a == 'E':
            n = i + 1
            return n if (i % self.size) != (self.size - 1) else i

        if a == 'W':
            n = i - 1
            return n if ((i % self.size) != 0) else i

        raise ValueError("unknown action")


def build_rewards(num_squares: int):
    """Build rewards for the grid world with the given number of squares.

    Note that the vector returned is an expected reward computed as

    R'_s = Expectation[R_{t+1} | S_t = s].

    However, as there is no discount factor and all rewards for all transitions
    are just -1, the expected reward is -1 for all transitions out of all
    non-terminal states and 0 for transitions out of terminal states.
    """
    rewards = np.full((num_squares), -1.0)
    rewards[0] = rewards[-1] = 0
    rewards.shape = (num_squares, 1)  # make it a column vector
    return rewards


def build_state_transition_matrix(
        gw: GridWorld, policy: typing.Dict[str, float]):
    """Build state transition matrix for the given policy.

    To clarify what this is, we set up the following notation:
    > A is the set of actions with cardinality |A|.
    > S is the set of states with cardinality |S|.
    > \\pi(a|s) = Prob[action = a | state = s] for a \\in A, s \\in S.
    > P^a is the state transition probability matrix for fixed action a.
        It has dimension (|S|x|S|). P^a(s1,s2) is the probability of
        transitioning from state s1 to state s2 (s1 being the row index and
        s2 being the column index). In our case, for fixed a, P^a(s1,s2) is
        1 if s2 is the neighbor of s1 for action a and 0 otherwise.

    As the given policy depends only on the action and is independent of
    the state s, \\pi(a|s) is just a scalar value for fixed a. To clarify
    this independence of state, we can just denote the policy as \\pi(a).

    We want to compute and return P = \\sum_{a \\in A} \\pi(a) P^a.

    Args:
        gd: Grid world on which the the policy is to be evaluated.
        policy: (action -> probability) map to evaluate, i.e. find the
            state-value function for.
    """
    mat = np.zeros((gw.num_squares, gw.num_squares))
    for i in range(gw.num_squares):
        for action in policy:
            neighbor = gw.neighbor(i, action)
            mat[i][neighbor] += policy[action]
    return mat


def run(
        n: int,
        policy: typing.Dict[str, float],
        iterations: int = 1000,
        atol: float = 0.001,):
    """Evalute the value function of the given policy on a nxn grid world.

    The grid world is a nxn matrix with the top-left and bottom-right suares as
    terminal states. The squares are numbered from left to right and row by row
    with the top-left square numbered as 0 and the bottom-right square numbered
    as (n*n - 1).

    An agent in the grid world can take 1 step to a neighboring square in the
    (N,E,S,W) direction.

    Args:
        n: number of squares on 1 side of the grid world.
        policy: (action -> probability) dict for which the state-value function
            is to be found.

    Returns:
        Matrix with the state-value function on each square of the grid world.
    """
    gw = GridWorld(size=n)
    rewards = build_rewards(gw.num_squares)
    log.debug(f"expected rewards: {rewards}")

    stm = build_state_transition_matrix(gw, policy)
    log.debug(f"state transition matrix:\n{stm}")

    prev_values = np.zeros((gw.num_squares))
    prev_values.shape = (gw.num_squares, 1)  # make it a column vector
    for i in range(iterations):
        values = rewards + np.matmul(stm, prev_values)
        max_diff = np.absolute(values - prev_values).max()
        if max_diff <= atol:
            log.info(f"num iterations for tol {atol}: {i+1}")
            break
        prev_values = values
        if i%10 == 0:
            log.info(f"iteration {i} {np.round(values.reshape(n, n), 1)}")

    log.info(f"final max_diff: {np.round(max_diff,3)}")
    return np.round(values.reshape(n, n), 1)


def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s--: %(message)s',
                        level=logging.INFO)

    # avoid line wrapping when printing numpy objects
    np.set_printoptions(linewidth=1000)  # Using integer instead of np.inf

    policy = {
        'N': 0.25,
        'S': 0.25,
        'E': 0.25,
        'W': 0.25,
    }
    final_value = run(n=4, policy=policy)
    log.info(f"final state value function:\n{final_value}")


if __name__ == '__main__':
    main()
