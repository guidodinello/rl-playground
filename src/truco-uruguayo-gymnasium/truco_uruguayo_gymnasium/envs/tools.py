import functools
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


def get_action(_id: int) -> str:
    if _id < 3:
        return f"Juega carta {_id+1} de la mano -> {get_card(_id+1)}"
    elif _id < 7:
        return f"Grita -> {get_bet(_id-3, 'Truco')}"
    elif _id < 10:
        return f"Toca -> {get_bet(_id-7, 'Envido')}"
    elif _id < 13:
        return f"Canta -> {get_bet(_id-10, 'Flor')}"
    elif _id < 17:
        return f"Responde -> {['Quiero','NoQuiero','MeVoy','SonBuenas'][_id-13]}"
    elif _id < 39:
        return f"Responde -> Tengo {_id-17} Puntos de Flor"
    elif _id < 73:
        return f"Responde -> Tengo {_id-39} Puntos de Envido"
    else:
        raise ValueError(f"Invalid action id {_id}")


def compose(*functions):
    def inner(arg, **kwargs):
        fns = list(reversed(functions))
        result = arg
        fns[0] = functools.partial(fns[0], **kwargs)
        for func in fns:
            result = func(result)
        return result

    return inner
