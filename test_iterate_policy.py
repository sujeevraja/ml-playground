#!/usr/bin/env python3

import iterate_policy as ip
import numpy as np
import unittest


class TestPolicyIteration(unittest.TestCase):
    def test_terminal_state_neighbors(self):
        policy_iteration = ip.PolicyIteration(4)
        for a in policy_iteration.actions:
            self.assertEqual(0, policy_iteration.get_neighbor(0, a))
            self.assertEqual(15, policy_iteration.get_neighbor(15, a))

    def test_middle_square_neighbors(self):
        policy_iteration = ip.PolicyIteration(4)
        self.assertEqual(2, policy_iteration.get_neighbor(6, 'N'))
        self.assertEqual(10, policy_iteration.get_neighbor(6, 'S'))
        self.assertEqual(7, policy_iteration.get_neighbor(6, 'E'))
        self.assertEqual(5, policy_iteration.get_neighbor(6, 'W'))

    def test_top_border_square_neighbors(self):
        policy_iteration = ip.PolicyIteration(4)
        self.assertEqual(1, policy_iteration.get_neighbor(1, 'N'))
        self.assertEqual(5, policy_iteration.get_neighbor(1, 'S'))
        self.assertEqual(2, policy_iteration.get_neighbor(1, 'E'))
        self.assertEqual(0, policy_iteration.get_neighbor(1, 'W'))

    def test_left_border_square_neighbors(self):
        policy_iteration = ip.PolicyIteration(4)
        self.assertEqual(4, policy_iteration.get_neighbor(8, 'N'))
        self.assertEqual(12, policy_iteration.get_neighbor(8, 'S'))
        self.assertEqual(9, policy_iteration.get_neighbor(8, 'E'))
        self.assertEqual(8, policy_iteration.get_neighbor(8, 'W'))

    def test_right_border_square_neighbors(self):
        policy_iteration = ip.PolicyIteration(4)
        self.assertEqual(7, policy_iteration.get_neighbor(11, 'N'))
        self.assertEqual(15, policy_iteration.get_neighbor(11, 'S'))
        self.assertEqual(11, policy_iteration.get_neighbor(11, 'E'))
        self.assertEqual(10, policy_iteration.get_neighbor(11, 'W'))

    def test_state_transition_matrix(self):
        """Check that all rows of the state transition matrix sum to 1."""
        policy_iteration = ip.PolicyIteration(4)
        policy = {a: 0.25 for a in policy_iteration.actions}
        stm = policy_iteration.build_state_transition_matrix(policy)
        self.assertTrue(np.allclose(np.full((1), 1.0), np.sum(stm, axis=1)))


if __name__ == '__main__':
    unittest.main()
