import random

import pytest

from guandan_engine.core import (
    BIG_JOKER_RANK,
    Card,
    HandState,
    Pattern,
    TrickState,
    TributePlan,
    apply_move,
    can_beat,
    classify_pattern,
    compute_tribute,
    generate_legal_moves,
    parse_cards,
    sort_cards,
    start_hand,
    update_levels_after_hand,
)


def make_state(trump_level, hands, current_player):
    trick = TrickState(leader=current_player, last_player=None, pattern=None, passes=(False, False, False, False))
    return HandState(
        trump_level=trump_level,
        hands=tuple(tuple(hand) for hand in hands),
        current_player=current_player,
        trick=trick,
        finished_order=tuple(),
    )


def test_classify_wildcard_straight():
    cards = parse_cards(["S2", "H3", "C4", "S6", "H10"])
    state = make_state(10, (tuple(),) * 4, 0)
    pattern = classify_pattern(cards, state)
    assert pattern is not None
    assert pattern.type == "STRAIGHT"
    assert pattern.main_rank == 6


def test_can_beat_bomb_hierarchy():
    state = make_state(10, (tuple(),) * 4, 0)
    bomb4 = classify_pattern(parse_cards(["S9", "H9", "C9", "D9"]), state)
    bomb5 = classify_pattern(parse_cards(["S8", "H8", "C8", "D8", "S8"]), state)
    assert bomb4 is not None and bomb5 is not None
    assert can_beat(bomb5, bomb4, state)
    assert not can_beat(bomb4, bomb5, state)


def test_generate_legal_moves_includes_consecutive_pairs():
    cards = [
        Card(rank=3, suit=0, card_id=0),
        Card(rank=3, suit=2, card_id=1),
        Card(rank=4, suit=0, card_id=2),
        Card(rank=4, suit=2, card_id=3),
        Card(rank=5, suit=0, card_id=4),
        Card(rank=5, suit=2, card_id=5),
    ]
    hands = (tuple(cards), tuple(), tuple(), tuple())
    state = make_state(10, hands, 0)
    moves = generate_legal_moves(0, state)
    found = False
    for combo in moves:
        pattern = classify_pattern(combo, state)
        if pattern and pattern.type == "THREE_CONSECUTIVE_PAIRS":
            found = True
            break
    assert found


def test_apply_move_and_trick_reset():
    deck = list(range(108))
    random.shuffle(deck)
    cards = [
        Card(rank=5, suit=0, card_id=0),
        Card(rank=6, suit=0, card_id=1),
        Card(rank=7, suit=0, card_id=2),
        Card(rank=8, suit=0, card_id=3),
        Card(rank=9, suit=0, card_id=4),
    ]
    hands = (tuple(cards), tuple(), tuple(), tuple())
    state = make_state(10, hands, 0)
    move = (cards[0],)
    state = apply_move(0, move, state)
    assert cards[0] not in state.hands[0]
    assert state.trick.pattern is not None


def test_compute_tribute_double_down():
    finished = [1, 3, 0, 2]
    hands = [
        sort_cards(parse_cards(["S9", "H8", "D7"]), 10),
        sort_cards(parse_cards(["S5", "S6", "S7", "S8", "S9"]), 10),
        sort_cards(parse_cards(["S10", "SJ", "SQ"]), 10),
        sort_cards(parse_cards(["H2", "H3", "H4"]), 10),
    ]
    plan = compute_tribute(finished, hands, 10)
    assert isinstance(plan, TributePlan)
    assert not plan.anti_tribute
    assert len(plan.exchanges) == 2
    assert plan.leader in {0, 2}


def test_anti_tribute_trigger():
    finished = [1, 3, 0, 2]
    hands = [
        sort_cards(parse_cards(["BJ", "BJ", "S9"]), 10),
        sort_cards(parse_cards(["S5", "S6"]), 10),
        sort_cards(parse_cards(["SJ", "SJ", "S8"]), 10),
        sort_cards(parse_cards(["H2", "H3"]), 10),
    ]
    plan = compute_tribute(finished, hands, 10)
    assert plan.anti_tribute
    assert not plan.exchanges


def test_update_levels_logic():
    finished = [0, 2, 1, 3]
    result = update_levels_after_hand(finished, current_even=5, current_odd=6)
    assert result.new_even == 8
    assert result.new_odd == 6


