#!/usr/bin/env python3

"""
Run this script to see an example of policy iteration to find the optimal
policy. This is based on example 4.2 from the following book:

Sutton, Richard S., and Andrew G. Barto.
Reinforcement learning: An introduction.
MIT press, 2018.

Problem statement:
Jack manages two locations for a nationwide car rental company. Each day, some
number of customers arrive at each location to rent cars. If Jack has a car
available, he rents it out and is credited $10 by the national company. If he is
out of cars at that location, then the business is lost. Cars become available
for renting the day after they are returned. To help ensure that cars are
available where they are needed, Jack can move them between the two locations
overnight, at a cost of $2 per car moved.  We assume that the number of cars
requested and returned at each location are Poisson random variables, meaning
that

Prob[X = n] = (lambda ** n) * (e ** (-lambda)) / n!

where lambda is the expected number. Suppose lambda is 3 and 4 for rental
requests at the first and second locations and 3 and 2 for returns. To simplify
the problem slightly, we assume that there can be no more than 20 cars at each
location (any additional cars are returned to the nationwide company, and thus
disappear from the problem) and a maximum of five cars can be moved from one
location to the other in one night. We take the discount rate to be  = 0.9 and
formulate this as a continuing finite MDP, where the time steps are days, the
state is the number of cars at each location at the end of the day, and the
actions are the net numbers of cars moved between the two locations overnight.
"""

import logging
import numpy as np
import typing

log = logging.getLogger(__name__)


class RentalCarEnv:
    def __init__(self):
        # Two locations, each with 0-20 cars
        self.state = [0, 0]
    
    def step(self, action: int):
        """
        Move cars between locations

        action: net number of cars moved from location 0 to location 1

        For example, say the state is [5,10] meaning that there are 5 cars at
         location 0 and 10 at location 1. If action is 2, then 2 cars are moved
         from location 0 to location 1, and the new state is [3, 12]. If the
         action is -2, then 2 cars are moved from location 1 to location 0, and
         the new state is [7, 8].
        
         Note that the number of cars at each location can never exceed 20.
        """
        self.state[0] = max(0, self.state[0] + action)
        self.state[1] = max(0, self.state[1] - action)
        for s in self.state:


def main():
    log.info("Starting policy iteration")
    env = RentalCarEnv(num_cars=20, max_move=5, discount=0.9)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)s--: %(message)s',
                        level=logging.INFO)
    main()
