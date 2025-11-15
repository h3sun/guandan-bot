import argparse
import random
from typing import List, Tuple

from .core import (
    HandState,
    Pattern,
    TrickState,
    build_double_deck,
    can_beat,
    classify_pattern,
    generate_legal_moves,
    parse_cards,
    start_hand,
)


def _hand_state_for_level(trump_level: int) -> HandState:
    empty = tuple()
    trick = TrickState(leader=0, last_player=None, pattern=None, passes=(False, False, False, False))
    return HandState(
        trump_level=trump_level,
        hands=(empty, empty, empty, empty),
        current_player=0,
        trick=trick,
        finished_order=tuple(),
    )


def _format_card(card) -> str:
    suit_map = {0: "S", 1: "H", 2: "C", 3: "D", 4: "J"}
    rank_map = {
        11: "J",
        12: "Q",
        13: "K",
        14: "A",
        15: "SJ",
        16: "BJ",
    }
    suit = suit_map[card.suit]
    rank = rank_map.get(card.rank, str(card.rank))
    if card.suit == 4:
        return rank
    return f"{suit}{rank}"


def _print_pattern(label: str, pattern: Pattern) -> None:
    card_text = " ".join(_format_card(card) for card in pattern.cards)
    print(f"{label}: {pattern.type} main={pattern.main_rank} len={pattern.length} cards={card_text}")


def run_demo(trump_level: int, seed: int) -> None:
    deck = list(build_double_deck())
    rng = random.Random(seed)
    rng.shuffle(deck)
    state = start_hand(trump_level, deck, leader=0)
    player_cards = state.hands[0]
    print(f"Player 0 hand ({len(player_cards)} cards):")
    print(" ".join(_format_card(card) for card in player_cards))
    moves = generate_legal_moves(0, state)
    print(f"Total legal openings: {len(moves)}")
    counts = {}
    for combo in moves:
        pattern = classify_pattern(combo, state)
        if pattern is None:
            continue
        counts[pattern.type] = counts.get(pattern.type, 0) + 1
    for pattern_type, count in sorted(counts.items()):
        print(f"  {pattern_type}: {count}")


def classify_cli(cards_text: str, trump_level: int, prev_text: str) -> None:
    cards = parse_cards(_split_tokens(cards_text))
    dummy_state = _hand_state_for_level(trump_level)
    pattern = classify_pattern(cards, dummy_state)
    if pattern is None:
        print("Pattern invalid")
    else:
        _print_pattern("Current", pattern)
    if prev_text:
        prev_cards = parse_cards(_split_tokens(prev_text))
        prev_pattern = classify_pattern(prev_cards, dummy_state)
        if prev_pattern is None:
            print("Previous pattern invalid")
        elif pattern is not None:
            result = can_beat(pattern, prev_pattern, dummy_state)
            status = "beats" if result else "does NOT beat"
            _print_pattern("Previous", prev_pattern)
            print(f"Current pattern {status} the previous pattern.")


def _split_tokens(text: str) -> List[str]:
    clean = text.replace(",", " ").split()
    return clean


def main() -> None:
    parser = argparse.ArgumentParser(description="Guandan engine CLI helper")
    parser.add_argument("--trump-level", type=int, required=True, help="Current level rank (2-14)")
    parser.add_argument("--cards", type=str, help="Comma or space separated cards like 'H10 S10 SJ BJ'")
    parser.add_argument("--previous", type=str, help="Cards of previous pattern for comparison")
    parser.add_argument("--demo", action="store_true", help="Deal random hand and list legal openings")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for demo")
    args = parser.parse_args()
    if args.cards:
        classify_cli(args.cards, args.trump_level, args.previous or "")
    if args.demo:
        run_demo(args.trump_level, args.seed)
    if not args.cards and not args.demo:
        parser.error("Provide --cards or --demo")


if __name__ == "__main__":
    main()

