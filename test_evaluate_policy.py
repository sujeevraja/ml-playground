#!/usr/bin/env python3

import evaluate_policy as ep
import numpy as np
import unittest


class TestGridWorldCreation(unittest.TestCase):
    def test_terminal_state_neighbors(self):
        gw = ep.GridWorld(size=4)
        for a in ['N', 'S', 'E', 'W']:
            self.assertEqual(0, gw.neighbor(0, a))
            self.assertEqual(15, gw.neighbor(15, a))

    def test_middle_square_neighbors(self):
        gw = ep.GridWorld(size=4)
        self.assertEqual(2, gw.neighbor(6, 'N'))
        self.assertEqual(10, gw.neighbor(6, 'S'))
        self.assertEqual(7, gw.neighbor(6, 'E'))
        self.assertEqual(5, gw.neighbor(6, 'W'))

    def test_top_border_square_neighbors(self):
        gw = ep.GridWorld(size=4)
        self.assertEqual(1, gw.neighbor(1, 'N'))
        self.assertEqual(5, gw.neighbor(1, 'S'))
        self.assertEqual(2, gw.neighbor(1, 'E'))
        self.assertEqual(0, gw.neighbor(1, 'W'))

    def test_left_border_square_neighbors(self):
        gw = ep.GridWorld(size=4)
        self.assertEqual(4, gw.neighbor(8, 'N'))
        self.assertEqual(12, gw.neighbor(8, 'S'))
        self.assertEqual(9, gw.neighbor(8, 'E'))
        self.assertEqual(8, gw.neighbor(8, 'W'))

    def test_right_border_square_neighbors(self):
        gw = ep.GridWorld(size=4)
        self.assertEqual(7, gw.neighbor(11, 'N'))
        self.assertEqual(15, gw.neighbor(11, 'S'))
        self.assertEqual(11, gw.neighbor(11, 'E'))
        self.assertEqual(10, gw.neighbor(11, 'W'))

    def test_state_transition_matrix(self):
        """Check that all rows of the state transition matrix sum to 1."""
        gw = ep.GridWorld(size=4)
        policy = {a: 0.25 for a in ['N', 'S', 'E', 'W']}
        stm = ep.build_state_transition_matrix(gw, policy)
        self.assertTrue(np.allclose(np.full((1), 1.0), np.sum(stm, axis=1)))


class TestPolicyEvaluation(unittest.TestCase):
    def test_1_iteration(self):
        pi = {'N': 0.25, 'S': 0.25, 'E': 0.25, 'W': 0.25, }
        v = np.array(
            [[0., - 1., - 1., - 1.],
             [-1., - 1., - 1., - 1.],
             [-1., - 1., - 1., - 1.],
             [-1., - 1., - 1.,  0.]])
        self.assertTrue(np.allclose(v, ep.run(4, pi, 1), atol=0.001))

    def test_2_iterations(self):
        pi = {'N': 0.25, 'S': 0.25, 'E': 0.25, 'W': 0.25, }
        v = np.array(
            [[0., - 1.8, - 2., - 2.],
             [-1.8, - 2., - 2., - 2.],
             [-2., - 2., - 2., - 1.8],
             [-2., - 2., - 1.8,  0.]])
        self.assertTrue(np.allclose(v, ep.run(4, pi, 2), atol=0.001))

    def test_3_iterations(self):
        pi = {'N': 0.25, 'S': 0.25, 'E': 0.25, 'W': 0.25, }
        v = np.array(
            [[0.,  -2.4, -2.9, -3.],
             [-2.4, -2.9, -3.,  -2.9],
             [-2.9, -3.,  -2.9, -2.4],
             [-3.,  -2.9, -2.4,  0.]])
        self.assertTrue(np.allclose(v, ep.run(4, pi, 3), atol=0.01))

    def test_10_iterations(self):
        pi = {'N': 0.25, 'S': 0.25, 'E': 0.25, 'W': 0.25, }
        v = np.array(
            [[0.,  -6.1, -8.4, -9.],
             [-6.1, -7.7, -8.4, -8.4],
             [-8.4, -8.4, -7.7, -6.1],
             [-9.,  -8.4, -6.1,  0.]])
        self.assertTrue(np.allclose(v, ep.run(4, pi, 10), atol=0.1))

    def test_convergence(self):
        pi = {'N': 0.25, 'S': 0.25, 'E': 0.25, 'W': 0.25, }
        v = np.array(
            [[0., -14., -20., -22.],
             [-14., -18., -20., -20.],
             [-20., -20., -18., -14.],
             [-22., -20., -14.,   0.]])
        self.assertTrue(np.allclose(v, ep.run(4, pi), atol=0.001))


if __name__ == '__main__':
    unittest.main()
