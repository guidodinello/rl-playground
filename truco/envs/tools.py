from typing import Literal, NamedTuple, NewType

ROUNDS_PER_HAND = 3
CARDS_PER_ROUND = 3
MATCH_POINTS = 20
ENVIDO_POINTS = 2


Player = NewType("Player", int)  # [0.. n_players-1]
Team = NewType("Team", int)  # [0, 1]
Card = NewType("Card", int)  # [0,1..40] # 0 special sentinel value
Bet = NewType("Bet", int)  # mapping 0->truco or 0->envido
Play = NamedTuple(
    "Play", [("player", Player), ("card", Card)]
)  # player X played card Y. juan jugo 4 de espada
History = NamedTuple(
    "History", [("player", Player), ("bet", Bet)]
)  # player X did bet Y. juan grito truco

BetType = Literal["Truco", "Envido", "Flor"]


def get_card(_id: int) -> str:
    if _id == 0:
        return "Inval_id"

    suit_index, rank_index = divmod(_id - 1, 10)

    suit = ["Oro", "Copa", "Espada", "Basto"][suit_index]
    rank = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12][rank_index]
    return f"{rank} de {suit}"


def get_bet(_id: int, _type: BetType) -> str:
    if _type == "Truco":
        return ["Truco", "Retruco", "ValeCuatro"][_id]
    elif _type == "Envido":
        return ["Envido", "RealEnvido", "FaltaEnvido"][_id]
    elif _type == "Flor":
        return ["LaMiaEsFlor", "ConFlorEnvido", "ContraFlorAlResto"][_id]


def get_player_team(_id: int) -> str:
    return ["Us", "Them"][_id % 2]
