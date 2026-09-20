__credits__ = ["Guido Dinello"]

import logging
import random
from functools import partial
from itertools import cycle
from typing import Dict, List, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from truco_uruguayo_gymnasium.envs.tools import (
    CARDS_PER_ROUND,
    ENVIDO_POINTS,
    MATCH_POINTS,
    ROUNDS_PER_HAND,
    History,
    Play,
    compose,
)

uint8_arange = partial(np.arange, dtype=np.uint8)
uint8_multidiscrete = partial(spaces.MultiDiscrete, dtype=np.uint8)
seq_multidiscrete = compose(spaces.Sequence, uint8_multidiscrete)


class TrucoUruguayoEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, n_players=2):
        self.n_players = n_players

        self.hands = [
            [p, p + self.n_players, p + 2 * self.n_players]
            for p in range(self.n_players)
        ]

        self.masks = {
            "cards_played": (uint8_arange(ROUNDS_PER_HAND * self.n_players), None),
            # "cards_in_hand": (),
            # "score": (),
            # "previous_round_winner": (),
            ##
            "envido_history": (uint8_arange(MATCH_POINTS / ENVIDO_POINTS), None),
            "envido_players_known_points": (uint8_arange(self.n_players), None),
            # "envido_cumbet": (),
            # "envido_points": (),
            ##
            "flor_history": (uint8_arange(4), None),
            "flor_players_known_points": (uint8_arange(self.n_players), None),
            # "flor_cumbet": (),
            # "flor_points": (),
            ##
            "truco_history": (uint8_arange(4), None),
            # "truco_state": (),
        }
        envido = {
            # historial de toques, player X dijo envido y player Y reviro con real, etc.
            # (0: envido, 1: real envido, 2: falta envido)
            # [
            #  [0, 0],  # player 1 dijo envido,
            #  [1, 1],  # player 2 reviro con real envido,
            #  [0, 2],  # player 1 reviro con falta envido,
            # ]
            "envido_history": seq_multidiscrete([self.n_players, 3]),
            # already announced points (X player, Y points) per player.
            # minimo: tres 4s ninguno de la muestra = 4 puntos. maximo: 37 puntos.
            "envido_players_known_points": seq_multidiscrete(
                [self.n_players, 37], start=[0, 4]
            ),
            # redundant but useful for the agent
            "envido_cumbet": spaces.Discrete(MATCH_POINTS),  # bet
            "envido_points": spaces.Discrete(37, start=4),  # agent calculated points
        }

        flor = {
            # (0: la mia es flor, 1: con flor envido, 2: contra flor al resto)
            #
            "flor_history": seq_multidiscrete([self.n_players, 3]),
            # already announced points (X player, Y points) per player.
            # minimo: derecha 1,2y3 = 26 puntos. maximo: 2,4y5 de la muestra = 47 puntos.
            "flor_players_known_points": seq_multidiscrete(
                [self.n_players, 47], start=[0, 26]
            ),
            # redundant but useful for the agent
            "flor_cumbet": spaces.Discrete(MATCH_POINTS),  # bet
            "flor_points": spaces.Discrete(47, start=26),  # agent calculated points
        }
        truco = {
            "truco_history": seq_multidiscrete(
                [self.n_players, 3]
            ),  # obs: capaz podria agregar la ronda al history de truco para saber en que ronda gritaron
            # already announced TrucoState (Nothing|Truco|Retruco|ValeCuatro)
            "truco_state": spaces.Discrete(4),
        }

        self.observation_space = spaces.Dict(
            {
                # cards already played in this hand, including the current round.
                # 40 cards + 1 (0= todavia no se jugo).
                # i care to know which card was played by which player?
                "cards_played": seq_multidiscrete([self.n_players, 41]),
                # 40 posible cards + 1 (0=ya la jugue)
                "cards_in_hand": uint8_multidiscrete([41] * CARDS_PER_ROUND),
                "score": spaces.Tuple(
                    [spaces.Discrete(MATCH_POINTS), spaces.Discrete(MATCH_POINTS)]
                ),
                # 0: no se jugo, 1: empate, 2: equipo 1, 3: equipo 2
                "previous_round_winner": spaces.Discrete(4),  # dudosa esta
            }
            | envido
            | flor
            | truco
        )

        # # calculate the observation space shape
        # self.observation_space.set_attr(
        #     "shape",
        #     {
        #         key: (value.shape[0],) if value.shape else ()
        #         for key, value in self.observation_space.spaces.items()
        #     },
        # )

        self.action_space = spaces.Discrete(71)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _initialize_state(self):
        self.game_state = {"score": [0, 0]}
        self.hand_state: Dict[str, Union[int, np.ndarray]] = {
            "round_winner_history": [],
            "deck": np.arange(40),
        }
        self.round_state: Dict[str, Union[Play, int, History, List[int]]] = {
            "cards_played": [],
            "truco_state": 0,
            "truco_history": [],
            "envido_history": [],
            "envido_players_known_points": [],
            "envido_cumbet": 0,
            "envido_points": 0,
            "flor_history": [],
            "flor_players_known_points": [],
            "flor_cumbet": 0,
            "flor_points": 0,
        }
        turns = np.random.permutation(np.arange(self.n_players))

        self.dealer = cycle(turns)
        self.player_turn = next(
            cycle(turns)
        )  # first player to play is the first rightmost to the dealer
        self.team_turn = np.random.choice([0, 1])  # dudoso

    def _finish_turn(self) -> int:
        """
        Should be called after each player plays a card.
        Returns the index of the next player to perform an action.
        """
        return next(self.player_turn)

    def _finish_round(self, winner):
        """Should be called after all players played a card."""
        # update hand_state
        self.hand_state["round_winner_history"].append(winner)
        # reset round_state
        self.round_state = {
            "cards_played": [],
            ##
            "truco_state": 0,
            "truco_history": [],
            ##
            "envido_history": [],
            "envido_players_known_points": [],
            "envido_cumbet": 0,  # posiblemente termine eliminando los cumbet del state (no del observation_space)
            "envido_points": 0,
            ##
            "flor_history": [],
            "flor_players_known_points": [],
            "flor_cumbet": 0,
            "flor_points": 0,
        }

    def _finish_hand(self, points: Tuple[int, int]) -> int:
        """
        Should be called after all players played their 3 cards.
        Returns the index of the player who must deal the new hand.
        """
        # update game_state
        self.game_state["score"] = [
            sum(x) for x in zip(self.game_state["score"], points)
        ]
        # reset hand_state
        self.hand_state["round_winner_history"] = []
        random.shuffle(self.hand_state["deck"])
        return next(self.dealer)

    def calculate_reward(self) -> int:
        """
        Should be called after every player played its 3 cards. same as _finish_hand
        """
        # winner_team_points = 0
        # if self.round_state["envido_history"] != []:
        #     # get player that won the envido. es el ultimo de envido_history
        #     winner_player = self.round_state["envido_history"][-1][0]
        #     # get team of the winner player
        #     winner_team = winner_player % 2  # revisar
        #     winner_team_points = self.round_state["envido_cumbet"]
        # elif self.round_state["flor_history"] != []:
        #     winner_team = self.round_state["flor_history"][-1][0] % 2
        #     winner_team_points = self.round_state["flor_cumbet"]
        # else:
        #     logging.info("No se toco ni canto")

        # team that won the most rounds, wins the hand.

        return 1  # our_team_points - opponent_team_points

    def _get_obs(self):
        obs = {**self.round_state, **self.game_state}
        history = self.hand_state["round_winner_history"]
        if not history:
            obs["previous_round_winner"] = 0
        else:
            obs["previous_round_winner"] = history[
                -1
            ]  # TODO: calcular bien. hayq ue ver si gano un equipo o hubo empate, fijarse las cartas jugadas
        obs["cards_in_hand"] = self.hand_state["deck"][self.hands[self.player_turn]]
        return obs

    def _get_info(self):
        return {"game_state": self.game_state}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._initialize_state()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        log_action(action)

        # return observation_space as result of action
        # reward
        reward = self.calculate_reward()
        observation = self._get_obs()
        info = self._get_info()
        terminated = any(score >= MATCH_POINTS for score in self.game_state["score"])

        if self.render_mode == "human":
            self._render_frame()

        self._finish_turn()
        return observation, reward, terminated, False, info

    def opponent_policy(self):
        ## TODO: probably should be injecting an agent instead

        # Implement a simple rule-based policy for the opponent
        # For example, always play the first available card
        for i in range(CARDS_PER_ROUND):
            if self.hand_state["deck"][self.dealer][i] != 41:
                self._finish_turn()
                return i

        assert False, "This should never happen"

    def render(self):
        raise NotImplementedError

    def _render_frame(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


def log_action(action: int):
    if action < 3:
        logging.info("Playing %s card", action)

    elif action < 7:
        logging.info("Announcing %s Truco", action - 3)
    elif action < 10:
        logging.info("Announcing %s Envido", action - 7)
    elif action < 13:
        logging.info("Announcing %s Flor", action - 10)
    elif action < 16:
        logging.info("Responding %s Quiero", action - 13)
    elif action == 16:
        logging.info("Responding SonBuenas")
    elif action < 64:
        logging.info("Responding Tengo %s Puntos de Flor", action - 16)
    else:
        logging.info("Responding Tengo %s Puntos de Envido", action - 64)
