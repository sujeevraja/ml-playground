#!/usr/bin/env python3

"""
Run this script for examples on policy iteration.
"""

import logging

log = logging.getLogger(__name__)


def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s--: %(message)s',
                        level=logging.INFO)
    log.info("Hello, world!")


if __name__ == '__main__':
    main()
