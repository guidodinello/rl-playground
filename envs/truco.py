from ast import Tuple


import gym
import numpy as np
from gym import spaces

ROUNDS_PER_HAND = 3
CARDS_PER_ROUND = 3
MATCH_POINTS = 20
ENVIDO_POINTS = 2


class TrucoUruguayo(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.n_players = 2

        self.hands = [
            [p, p + self.n_players, p + 2 * self.n_players]
            for p in range(self.n_players)
        ]
        self.game_state = {"score": [0, 0]}
        self.hand_state = {
            "round_winner_history": [],
            "deck": np.arange(40),
        }
        self.round_state = {
            "truco_state": 0,
            "cards_played": [],
            "envido_history": [],
            "envido_players_known_points": [],
            "envido_cumbet": 0,
            "envido_points": 0,
            "flor_history": [],
            "flor_players_known_points": [],
            "truco_history": [],
        }

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
            }
        )

        # obs: capaz podria agregar la ronda al history de truco para saber en que ronda gritaron
        truco = spaces.Dict(
            {
                "truco_history": spaces.Sequence(
                    spaces.MultiDiscrete([self.n_players, 3], dtype=int),
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
                "previous_round_winner": spaces.Discrete([3], dtype=int),
            }
            | envido
            | flor
            | truco
        )

        # play card (1|2|3)
        # announce (Truco|Retruco|ValeCuatro)
        # announce (Envido|RealEnvido|FaltaEnvido)
        # announce (LaMiaEsFlor|ConFlorEnvido|ContraFlorAlResto)
        # response (Quiero|NoQuiero|MeVoy)
        # reponse (SonBuenas)
        # response (Tengo[47-26] Puntos de Flor)
        # response (Tengo[37-4] Puntos de Envido)
        self.action_space = spaces.Discrete(
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

    def _finish_round(self, winner):
        # update hand_state
        self.hand_state["round_winner_history"].append(winner)
        # reset round_state
        self.round_state = {
            "truco_state": 0,
            "cards_played": [],
            "envido_history": [],
            "envido_players_known_points": [],
            "envido_cumbet": 0,
            "envido_points": 0,
            "flor_history": [],
            "flor_players_known_points": [],
            "truco_history": [],
        }

    def _finish_hand(self, points: Tuple[int, int]):
        # update game_state
        self.game_state["score"] = [
            sum(x) for x in zip(self.game_state["score"], points)
        ]
        # reset hand_state
        self.hand_state = {
            "round_winner_history": [],
            "deck": np.arange(40),
        }

    def calculate_reward(self):
        pass

    def _get_obs(self):
        pass

    def _get_info(self):
        return {
            "nadie": "sabe",
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        if action < 3:
            print(f"Playing {action} card")
        elif action < 7:
            print(f"Announcing {action - 3} Truco")
        elif action < 10:
            print(f"Announcing {action - 7} Envido")
        elif action < 13:
            print(f"Announcing {action - 10} Flor")
        elif action < 16:
            print(f"Responding {action - 13} Quiero")
        elif action == 16:
            print("Responding SonBuenas")
        elif action < 64:
            print(f"Responding Tengo {action - 16} Puntos de Flor")
        else:
            print(f"Responding Tengo {action - 64} Puntos de Envido")

        # return observation_space as result of action
        # reward
        reward = self.calculate_reward()

        info = self._get_info()
        terminated = any(score >= MATCH_POINTS for score in self.state["game_score"])

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        pass

    def _render_frame(self):
        pass

    def close(self):
        pass
