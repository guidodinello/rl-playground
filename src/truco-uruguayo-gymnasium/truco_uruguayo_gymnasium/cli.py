"""Handles the initialization of the game through the command line interface."""

import argparse

from truco_uruguayo_gymnasium.tests.test_dqn import play as dqn_agent_env
from truco_uruguayo_gymnasium.tests.test_human import play as human_agent_env
from truco_uruguayo_gymnasium.tests.test_random import play as random_agent_env


def _get_args():
    """Parses the command line arguments and returns them."""
    parser = argparse.ArgumentParser(description=__doc__)

    # Argument for the mode of execution (human or random):
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="human",
        choices=["human", "random", "dqn"],
        help="The execution mode for the game.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="If set, the game will be executed without rendering it.",
    )

    return parser.parse_args()


def main():
    args = _get_args()

    if args.mode == "human":
        human_agent_env()
    elif args.mode == "random":
        random_agent_env(
            audio_on=(not args.quiet), render_mode="human" if not args.quiet else None
        )
    elif args.mode == "dqn":
        dqn_agent_env(
            audio_on=(not args.quiet), render_mode="human" if not args.quiet else None
        )
    else:
        print("Invalid mode!")
