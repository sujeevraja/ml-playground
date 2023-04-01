# ml-playground

A repository for me to play around with reinforcement learning algorithms and
tools.

## Usage

- Install `swig`. On macs, it can be done with `brew install swig`.
- Clone the repo and install dependencies with `pipenv install`.
- Open a shell with `pipenv shell` and run `./iterate_policy.py`.
- Run unit tests with `./test_iterate_policy.py -v`. 


## Files

- `iterate_policy.py`: File that does policy iteration on a 4-by-4 grid using
    the Bellman equation to find the optimal state value function.
    This code is tested by `test_iterate_policy.py`.