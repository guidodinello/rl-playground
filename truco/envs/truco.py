import logging
import random
from itertools import cycle
from typing import Dict, List, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from truco.envs.tools import (
    CARDS_PER_ROUND,
    ENVIDO_POINTS,
    MATCH_POINTS,
    ROUNDS_PER_HAND,
    History,
    Play,
)


class TrucoUruguayo(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, n_players=2):
        self.n_players = n_players

        self.hands = [
            [p, p + self.n_players, p + 2 * self.n_players]
            for p in range(self.n_players)
        ]

        envido = spaces.Dict(
            {
                # historial de toques, player X dijo envido y player Y reviro con real, etc.
                # (0: envido, 1: real envido, 2: falta envido)
                # [
                #  [0, 0],  # player 1 dijo envido,
                #  [1, 1],  # player 2 reviro con real envido,
                # ...
                # ]
                "envido_history": spaces.Sequence(
                    spaces.MultiDiscrete([self.n_players, 3], dtype=int),
                    maxlen=MATCH_POINTS / ENVIDO_POINTS,
                ),
                # already announced points (X player, Y points) per player.
                # minimo: tres 4s ninguno de la muestra = 4 puntos. maximo: 37 puntos.
                "envido_players_known_points": spaces.Sequence(
                    spaces.MultiDiscrete([self.n_players, 37 - 4], dtype=int),
                    maxlen=self.n_players,
                ),
                # redundant but useful for the agent
                "envido_cumbet": spaces.Discrete(MATCH_POINTS, dtype=int),  # bet
                "envido_points": spaces.Discrete(
                    37 - 4, dtype=int
                ),  # agent calculated points
            }
        )

        flor = spaces.Dict(
            {
                # (0: la mia es flor, 1: con flor envido, 2: contra flor al resto)
                "flor_history": spaces.Sequence(
                    spaces.MultiDiscrete([self.n_players, 3], dtype=int),
                    maxlen=3,  # no estoy seguro, pero asumo [player 1 dice la mia es flor, player 2 contest con flor envid y player 1 contesta contra flor al resto]
                ),
                # already announced points (X player, Y points) per player.
                # minimo: derecha 1,2y3 = 26 puntos. maximo: 2,4y5 de la muestra = 47 puntos.
                "flor_players_known_points": spaces.Sequence(
                    spaces.MultiDiscrete([self.n_players, 47 - 26], dtype=int),
                    maxlen=self.n_players,
                ),
                # redundant but useful for the agent
                "flor_cumbet": spaces.Discrete(MATCH_POINTS, dtype=int),  # bet
                "flor_points": spaces.Discrete(
                    47 - 26, dtype=int
                ),  # agent calculated points
            }
        )

        truco = spaces.Dict(
            {
                "truco_history": spaces.Sequence(
                    spaces.MultiDiscrete(
                        [self.n_players, 3], dtype=int
                    ),  # obs: capaz podria agregar la ronda al history de truco para saber en que ronda gritaron
                    maxlen=3,
                ),
                # already announced TrucoState (Nothing|Truco|Retruco|ValeCuatro)
                "truco_state": spaces.Discrete(4, dtype=int, default=0),
            }
        )

        self.observation_space = spaces.Dict(
            {
                # cards already played in this round, including the current hand.
                # 40 cards + 1 (? todavia no se jugo).
                # i care to know which card was played by which player.
                "cards_played": spaces.Sequence(
                    spaces.MultiDiscrete([self.n_players, 41], dtype=int, default=0),
                    maxlen=ROUNDS_PER_HAND * self.n_players,
                ),
                # 40 cards + 1 (ya la jugue?)
                "cards_in_hand": spaces.MultiDiscrete(
                    [41] * CARDS_PER_ROUND, dtype=int
                ),
                "game_score": spaces.MultiDiscrete([MATCH_POINTS] * 2, dtype=int),
                # 0: empate, 1: equipo 1, 2: equipo 2
                "previous_round_winner": spaces.Discrete([3], dtype=int),  # dudosa esta
            }
            | envido
            | flor
            | truco
        )

        self.action_space = spaces.Discrete(
            # play card (1|2|3)
            # announce (Truco|Retruco|ValeCuatro)
            # announce (Envido|RealEnvido|FaltaEnvido)
            # announce (LaMiaEsFlor|ConFlorEnvido|ContraFlorAlResto)
            # response (Quiero|NoQuiero|MeVoy)
            # reponse (SonBuenas)
            # response (Tengo[47-26] Puntos de Flor)
            # response (Tengo[37-4] Puntos de Envido)
            3  # play card (1|2|3)
            + 4  # announce (Truco|Retruco|ValeCuatro)
            + 3  # announce (Envido|RealEnvido|FaltaEnvido)
            + 3  # announce (LaMiaEsFlor|ConFlorEnvido|ContraFlorAlResto)
            + 3  # response (Quiero|NoQuiero|MeVoy)
            + 1  # reponse (SonBuenas)
            + (47 - 26)  # response (Tengo[47-26] Puntos de Flor)
            + (37 - 4)  # response (Tengo[37-4] Puntos de Envido)
        )

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
        return next(self.turn)

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
        pass

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
        if action < 3:
            logging.info(f"Playing {action} card")

        elif action < 7:
            logging.info(f"Announcing {action - 3} Truco")
        elif action < 10:
            logging.info(f"Announcing {action - 7} Envido")
        elif action < 13:
            logging.info(f"Announcing {action - 10} Flor")
        elif action < 16:
            logging.info(f"Responding {action - 13} Quiero")
        elif action == 16:
            logging.info("Responding SonBuenas")
        elif action < 64:
            logging.info(f"Responding Tengo {action - 16} Puntos de Flor")
        else:
            logging.info(f"Responding Tengo {action - 64} Puntos de Envido")

        # return observation_space as result of action
        # reward
        reward = self.calculate_reward()
        observation = self._get_obs()
        info = self._get_info()
        terminated = any(score >= MATCH_POINTS for score in self.state["game_score"])

        if self.render_mode == "human":
            self._render_frame()

        self._finish_turn()
        return observation, reward, terminated, False, info

    def opponent_policy(self):
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
        raise NotImplementedError

    def _render_frame(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
        raise NotImplementedError
