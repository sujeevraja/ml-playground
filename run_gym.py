#!/usr/bin/env python3

"""Run this script to see some ways to train RL models with gymnasium"""

import gymnasium as gym


def try_gym():
    env = gym.make("LunarLander-v2", render_mode="human")
    observation, info = env.reset()

    for _ in range(500):
        # agent policy that uses the observation and info
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()

    print("observation", observation)
    print("reward", reward)
    print("terminated", terminated)
    print("truncated", truncated)
    print("info", info)


if __name__ == '__main__':
    try_gym()
