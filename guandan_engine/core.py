from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


Suit = int  # 0: spade, 1: heart, 2: club, 3: diamond, 4: joker
Rank = int  # 2-14 normal ranks, 15 small joker, 16 big joker
PlayerID = int  # 0..3

HEART_SUIT: Suit = 1
JOKER_SUIT: Suit = 4
SMALL_JOKER_RANK: Rank = 15
BIG_JOKER_RANK: Rank = 16
PLAIN_SUITS: Tuple[Suit, Suit, Suit, Suit] = (0, 1, 2, 3)
PATTERN_TYPES: Tuple[str, ...] = (
    "FOUR_JOKERS",
    "BOMB_8",
    "BOMB_7",
    "BOMB_6",
    "BOMB_5",
    "BOMB_4",
    "STRAIGHT_FLUSH",
    "STRAIGHT",
    "THREE_CONSECUTIVE_PAIRS",
    "STEEL_PLATE",
    "TRIPLE_PLUS_PAIR",
    "TRIPLE",
    "PAIR",
    "SINGLE",
)
TRUMP_SEQUENCE_RANKS: Tuple[Rank, ...] = (BIG_JOKER_RANK, SMALL_JOKER_RANK)
MAX_HAND_CARDS: int = 27


@dataclass(frozen=True)
class Card:
    rank: Rank
    suit: Suit
    card_id: int


@dataclass(frozen=True)
class Pattern:
    type: str
    main_rank: Rank
    length: int
    cards: Tuple[Card, ...]
    uses_wildcard: bool


@dataclass(frozen=True)
class TrickState:
    leader: PlayerID
    last_player: Optional[PlayerID]
    pattern: Optional[Pattern]
    passes: Tuple[bool, bool, bool, bool]


@dataclass(frozen=True)
class HandState:
    trump_level: Rank
    hands: Tuple[Tuple[Card, ...], Tuple[Card, ...], Tuple[Card, ...], Tuple[Card, ...]]
    current_player: PlayerID
    trick: TrickState
    finished_order: Tuple[PlayerID, ...]


@dataclass(frozen=True)
class MatchState:
    level_even: Rank
    level_odd: Rank
    dealer: PlayerID
    hand: Optional[HandState]


@dataclass(frozen=True)
class TributeExchange:
    donor: PlayerID
    recipient: PlayerID
    tribute_card: Card
    refund_card: Optional[Card]


@dataclass(frozen=True)
class TributePlan:
    exchanges: Tuple[TributeExchange, ...]
    leader: PlayerID
    anti_tribute: bool


@dataclass(frozen=True)
class LevelUpdate:
    new_even: Rank
    new_odd: Rank
    winner_team: str
    passed_A: bool


def build_double_deck() -> Tuple[Card, ...]:
    cards: List[Card] = []
    card_id: int = 0
    for deck_index in range(2):
        for suit in PLAIN_SUITS:
            for rank in range(2, 15):
                cards.append(Card(rank=rank, suit=suit, card_id=card_id))
                card_id += 1
        cards.append(Card(rank=SMALL_JOKER_RANK, suit=JOKER_SUIT, card_id=card_id))
        card_id += 1
        cards.append(Card(rank=BIG_JOKER_RANK, suit=JOKER_SUIT, card_id=card_id))
        card_id += 1
    return tuple(cards)


def card_sort_key(card: Card, trump_level: Rank) -> Tuple[int, int]:
    return card_strength(card, trump_level), card.card_id


def sort_cards(cards: Sequence[Card], trump_level: Rank) -> List[Card]:
    return sorted(cards, key=lambda c: card_sort_key(c, trump_level), reverse=True)


def card_strength(card: Card, trump_level: Rank) -> int:
    if card.rank == BIG_JOKER_RANK:
        return 10_000
    if card.rank == SMALL_JOKER_RANK:
        return 9_000
    if card.rank == trump_level:
        if card.suit == HEART_SUIT:
            return 8_500
        return 8_000 + card.rank
    return card.rank


def is_wildcard(card: Card, trump_level: Rank) -> bool:
    return card.rank == trump_level and card.suit == HEART_SUIT


def rank_order_value(rank: Rank, trump_level: Rank) -> int:
    if rank == BIG_JOKER_RANK:
        return 1_000
    if rank == SMALL_JOKER_RANK:
        return 900
    if rank == trump_level:
        return 800
    return rank


def start_hand(trump_level: Rank, deck: Sequence[Card], leader: PlayerID) -> HandState:
    if len(deck) != 108:
        raise ValueError("Deck must contain 108 cards")
    hands: List[List[Card]] = [[], [], [], []]
    for idx, card in enumerate(deck):
        pid: PlayerID = idx % 4
        hands[pid].append(card)
    frozen_hands: Tuple[Tuple[Card, ...], Tuple[Card, ...], Tuple[Card, ...], Tuple[Card, ...]] = (
        tuple(sort_cards(hands[0], trump_level)),
        tuple(sort_cards(hands[1], trump_level)),
        tuple(sort_cards(hands[2], trump_level)),
        tuple(sort_cards(hands[3], trump_level)),
    )
    trick = TrickState(
        leader=leader,
        last_player=None,
        pattern=None,
        passes=(False, False, False, False),
    )
    return HandState(
        trump_level=trump_level,
        hands=frozen_hands,
        current_player=leader,
        trick=trick,
        finished_order=tuple(),
    )


def classify_pattern(cards: Sequence[Card], state: HandState) -> Optional[Pattern]:
    if not cards:
        return None
    normalized = tuple(sort_cards(cards, state.trump_level))
    length = len(normalized)
    wildcards = [card for card in normalized if is_wildcard(card, state.trump_level)]
    fixed_cards = [card for card in normalized if not is_wildcard(card, state.trump_level)]
    counts = _rank_counts(fixed_cards)
    if length == 4 and _is_four_jokers(normalized):
        return Pattern(
            type="FOUR_JOKERS",
            main_rank=BIG_JOKER_RANK,
            length=4,
            cards=normalized,
            uses_wildcard=False,
        )
    for target_len in range(8, 3, -1):
        if length == target_len and _can_form_same_rank(
            counts, len(wildcards), target_len
        ):
            main_rank = _same_rank_value(counts, len(wildcards), target_len, state.trump_level)
            return Pattern(
                type=f"BOMB_{target_len}",
                main_rank=main_rank,
                length=target_len,
                cards=normalized,
                uses_wildcard=length > len(fixed_cards),
            )
    if length == 5:
        straight_flush = _detect_sequence_pattern(
            normalized, state.trump_level, require_flush=True, pattern_type="STRAIGHT_FLUSH"
        )
        if straight_flush is not None:
            return straight_flush
    if length == 5:
        triple_plus_pair = _detect_triple_plus_pair(
            counts, len(wildcards), normalized, state.trump_level
        )
        if triple_plus_pair is not None:
            return triple_plus_pair
    if length == 5:
        straight_pattern = _detect_sequence_pattern(
            normalized, state.trump_level, require_flush=False, pattern_type="STRAIGHT"
        )
        if straight_pattern is not None:
            return straight_pattern
    if length == 6:
        triple_pairs = _detect_three_consecutive_pairs(
            normalized, state.trump_level, len(wildcards)
        )
        if triple_pairs is not None:
            return triple_pairs
        steel_plate = _detect_steel_plate(normalized, state.trump_level, len(wildcards))
        if steel_plate is not None:
            return steel_plate
    if length == 3 and _can_form_same_rank(counts, len(wildcards), 3):
        main_rank = _same_rank_value(counts, len(wildcards), 3, state.trump_level)
        return Pattern(
            type="TRIPLE",
            main_rank=main_rank,
            length=3,
            cards=normalized,
            uses_wildcard=length > len(fixed_cards),
        )
    if length == 2 and _can_form_same_rank(counts, len(wildcards), 2):
        main_rank = _same_rank_value(counts, len(wildcards), 2, state.trump_level)
        return Pattern(
            type="PAIR",
            main_rank=main_rank,
            length=2,
            cards=normalized,
            uses_wildcard=length > len(fixed_cards),
        )
    if length == 1:
        return Pattern(
            type="SINGLE",
            main_rank=normalized[0].rank,
            length=1,
            cards=normalized,
            uses_wildcard=is_wildcard(normalized[0], state.trump_level),
        )
    return None


def _is_four_jokers(cards: Tuple[Card, ...]) -> bool:
    ranks = {card.rank for card in cards}
    suits = {card.suit for card in cards}
    return len(cards) == 4 and ranks.issubset({SMALL_JOKER_RANK, BIG_JOKER_RANK}) and suits == {JOKER_SUIT}


def _rank_counts(cards: Sequence[Card]) -> Dict[Rank, int]:
    counts: Dict[Rank, int] = {}
    for card in cards:
        counts[card.rank] = counts.get(card.rank, 0) + 1
    return counts


def _can_form_same_rank(
    counts: Dict[Rank, int], wildcard_count: int, target_len: int
) -> bool:
    for rank, count in counts.items():
        if count + wildcard_count >= target_len and rank not in (SMALL_JOKER_RANK, BIG_JOKER_RANK):
            return True
    if wildcard_count >= target_len:
        return True
    return False


def _same_rank_value(
    counts: Dict[Rank, int], wildcard_count: int, target_len: int, trump_level: Rank
) -> Rank:
    best_rank: Optional[Rank] = None
    for rank, count in counts.items():
        if rank in (SMALL_JOKER_RANK, BIG_JOKER_RANK):
            continue
        if count + wildcard_count >= target_len:
            if best_rank is None or rank_order_value(rank, trump_level) > rank_order_value(
                best_rank, trump_level
            ):
                best_rank = rank
    if best_rank is not None:
        return best_rank
    return trump_level


def _detect_straight_flush(
    cards: Tuple[Card, ...],
    trump_level: Rank,
    counts: Dict[Rank, int],
    wildcards: Sequence[Card],
) -> Optional[Pattern]:
    suits = {card.suit for card in cards if not is_wildcard(card, trump_level)}
    if len(suits) > 1:
        return None
    target_suit: Optional[Suit] = None
    for card in cards:
        if not is_wildcard(card, trump_level) and card.suit != JOKER_SUIT:
            target_suit = card.suit
            break
    if target_suit is None:
        target_suit = HEART_SUIT
    return _detect_sequence_pattern(cards, trump_level, True, "STRAIGHT_FLUSH")


def _detect_sequence_pattern(
    cards: Tuple[Card, ...],
    trump_level: Rank,
    require_flush: bool,
    pattern_type: str,
) -> Optional[Pattern]:
    if len(cards) != 5:
        return None
    if any(card.rank in (SMALL_JOKER_RANK, BIG_JOKER_RANK) for card in cards):
        return None
    wildcards = [card for card in cards if is_wildcard(card, trump_level)]
    fixed = [card for card in cards if not is_wildcard(card, trump_level)]
    suits = {card.suit for card in fixed}
    target_suit: Optional[Suit] = None
    if require_flush:
        if any(card.suit == JOKER_SUIT for card in fixed):
            return None
        if len(suits) > 1:
            return None
        target_suit = suits.pop() if suits else HEART_SUIT
        for card in fixed:
            if card.suit != target_suit:
                return None
    ranks = [card.rank for card in fixed]
    if len(set(ranks)) != len(ranks):
        return None
    sequences = [
        (14, 2, 3, 4, 5),
        (2, 3, 4, 5, 6),
        (3, 4, 5, 6, 7),
        (4, 5, 6, 7, 8),
        (5, 6, 7, 8, 9),
        (6, 7, 8, 9, 10),
        (7, 8, 9, 10, 11),
        (8, 9, 10, 11, 12),
        (9, 10, 11, 12, 13),
        (10, 11, 12, 13, 14),
    ]
    for sequence in sequences:
        if all(rank in sequence for rank in ranks):
            missing = len(sequence) - len(ranks)
            if missing <= len(wildcards):
                main_rank = sequence[-1] if sequence != (14, 2, 3, 4, 5) else 5
                return Pattern(
                    type=pattern_type,
                    main_rank=main_rank,
                    length=5,
                    cards=cards,
                    uses_wildcard=len(wildcards) > 0,
                )
    return None


def _detect_three_consecutive_pairs(
    cards: Tuple[Card, ...],
    trump_level: Rank,
    wildcard_count: int,
) -> Optional[Pattern]:
    if len(cards) != 6:
        return None
    fixed = [card for card in cards if not is_wildcard(card, trump_level)]
    if any(card.rank in (SMALL_JOKER_RANK, BIG_JOKER_RANK) for card in fixed):
        return None
    counts = _rank_counts(fixed)
    sequences = [
        (14, 2, 3),
        (2, 3, 4),
        (3, 4, 5),
        (4, 5, 6),
        (5, 6, 7),
        (6, 7, 8),
        (7, 8, 9),
        (8, 9, 10),
        (9, 10, 11),
        (10, 11, 12),
        (11, 12, 13),
        (12, 13, 14),
    ]
    for sequence in sequences:
        if any(rank not in sequence for rank in counts if counts[rank] > 0):
            continue
        needed = sum(max(0, 2 - counts.get(rank, 0)) for rank in sequence)
        if needed <= wildcard_count:
            main_rank = sequence[-1] if sequence != (14, 2, 3) else 3
            return Pattern(
                type="THREE_CONSECUTIVE_PAIRS",
                main_rank=main_rank,
                length=6,
                cards=cards,
                uses_wildcard=wildcard_count > 0,
            )
    return None


def _detect_steel_plate(
    cards: Tuple[Card, ...],
    trump_level: Rank,
    wildcard_count: int,
) -> Optional[Pattern]:
    if len(cards) != 6:
        return None
    fixed = [card for card in cards if not is_wildcard(card, trump_level)]
    if any(card.rank in (SMALL_JOKER_RANK, BIG_JOKER_RANK) for card in fixed):
        return None
    counts = _rank_counts(fixed)
    sequences = [(rank, rank + 1) for rank in range(2, 14)]
    sequences.append((13, 14))
    for sequence in sequences:
        if any(rank not in sequence for rank in counts if counts[rank] > 0):
            continue
        needed = sum(max(0, 3 - counts.get(rank, 0)) for rank in sequence)
        if needed <= wildcard_count:
            return Pattern(
                type="STEEL_PLATE",
                main_rank=sequence[-1],
                length=6,
                cards=cards,
                uses_wildcard=wildcard_count > 0,
            )
    return None


def _detect_triple_plus_pair(
    counts: Dict[Rank, int],
    wildcard_count: int,
    cards: Tuple[Card, ...],
    trump_level: Rank,
) -> Optional[Pattern]:
    for triple_rank in range(2, 15):
        triple_have = counts.get(triple_rank, 0)
        need_triple = max(0, 3 - triple_have)
        if need_triple > wildcard_count:
            continue
        remaining = wildcard_count - need_triple
        for pair_rank in range(2, 17):
            if pair_rank == triple_rank:
                continue
            pair_have = counts.get(pair_rank, 0)
            need_pair = max(0, 2 - pair_have)
            if need_pair <= remaining:
                return Pattern(
                    type="TRIPLE_PLUS_PAIR",
                    main_rank=triple_rank,
                    length=5,
                    cards=cards,
                    uses_wildcard=wildcard_count > 0,
                )
    return None


def can_beat(new_pattern: Pattern, prev_pattern: Optional[Pattern], state: HandState) -> bool:
    if prev_pattern is None:
        return True
    if prev_pattern.type == "FOUR_JOKERS":
        return False
    if new_pattern.type == "FOUR_JOKERS":
        return True
    priority = _pattern_priority(new_pattern.type)
    prev_priority = _pattern_priority(prev_pattern.type)
    if priority != prev_priority:
        return priority > prev_priority
    if new_pattern.type.startswith("BOMB") and prev_pattern.type.startswith("BOMB"):
        if new_pattern.length != prev_pattern.length:
            return new_pattern.length > prev_pattern.length
    if new_pattern.type != prev_pattern.type or new_pattern.length != prev_pattern.length:
        return False
    return rank_order_value(new_pattern.main_rank, state.trump_level) > rank_order_value(
        prev_pattern.main_rank, state.trump_level
    )


def _pattern_priority(pattern_type: str) -> int:
    mapping = {
        "FOUR_JOKERS": 6,
        "BOMB_8": 5,
        "BOMB_7": 5,
        "BOMB_6": 5,
        "STRAIGHT_FLUSH": 4,
        "BOMB_5": 3,
        "BOMB_4": 2,
    }
    if pattern_type in mapping:
        return mapping[pattern_type]
    if pattern_type.startswith("BOMB"):
        return 1
    return 0


def generate_legal_moves(player: PlayerID, state: HandState) -> List[Tuple[Card, ...]]:
    hand_cards = list(state.hands[player])
    if not hand_cards:
        return []
    combos = _enumerate_all_patterns(hand_cards, state.trump_level)
    trick_pattern = state.trick.pattern
    if trick_pattern is None or state.trick.last_player is None:
        return [combo for combo, _ in combos]
    legal: List[Tuple[Card, ...]] = []
    for combo, pattern in combos:
        if can_beat(pattern, trick_pattern, state):
            legal.append(combo)
    return legal


def _enumerate_all_patterns(
    cards: Sequence[Card], trump_level: Rank
) -> List[Tuple[Tuple[Card, ...], Pattern]]:
    candidates: Dict[Tuple[int, ...], Pattern] = {}
    singles = [(card,) for card in cards]
    for combo in singles:
        pattern = _classify_simple_combo(combo, trump_level)
        if pattern is not None:
            candidates[_combo_key(combo)] = pattern
    rank_buckets: Dict[Rank, List[Card]] = {}
    wildcards: List[Card] = []
    for card in cards:
        if is_wildcard(card, trump_level):
            wildcards.append(card)
        else:
            rank_buckets.setdefault(card.rank, []).append(card)
    for target_len in range(2, 9):
        for rank in range(2, 17):
            combos = _build_rank_combos(rank, target_len, rank_buckets.get(rank, []), wildcards)
            for combo in combos:
                pattern = _classify_simple_combo(combo, trump_level)
                if pattern is not None:
                    candidates[_combo_key(combo)] = pattern
    sequence_combos = _build_sequence_combos(cards, trump_level)
    for combo in sequence_combos:
        pattern = _classify_simple_combo(combo, trump_level)
        if pattern is not None:
            candidates[_combo_key(combo)] = pattern
    pair_sequences = _build_consecutive_pairs_combos(cards, trump_level)
    for combo in pair_sequences:
        pattern = _classify_simple_combo(combo, trump_level)
        if pattern is not None:
            candidates[_combo_key(combo)] = pattern
    steel_sequences = _build_steel_plate_combos(cards, trump_level)
    for combo in steel_sequences:
        pattern = _classify_simple_combo(combo, trump_level)
        if pattern is not None:
            candidates[_combo_key(combo)] = pattern
    return [(combo_from_key(key, cards), pattern) for key, pattern in candidates.items()]


def _classify_simple_combo(combo: Tuple[Card, ...], trump_level: Rank) -> Optional[Pattern]:
    dummy_state = HandState(
        trump_level=trump_level,
        hands=(tuple(),) * 4,
        current_player=0,
        trick=TrickState(leader=0, last_player=None, pattern=None, passes=(False,) * 4),
        finished_order=tuple(),
    )
    return classify_pattern(combo, dummy_state)


def _combo_key(combo: Tuple[Card, ...]) -> Tuple[int, ...]:
    return tuple(sorted(card.card_id for card in combo))


def combo_from_key(key: Tuple[int, ...], cards: Sequence[Card]) -> Tuple[Card, ...]:
    card_map = {card.card_id: card for card in cards}
    return tuple(card_map[cid] for cid in key)


def _build_rank_combos(
    rank: Rank,
    target_len: int,
    cards: Sequence[Card],
    wildcards: Sequence[Card],
) -> List[Tuple[Card, ...]]:
    combos: List[Tuple[Card, ...]] = []
    available = list(cards)
    if len(available) >= target_len:
        for selection in combinations(available, target_len):
            combos.append(selection)
    for wild_len in range(1, min(target_len, len(wildcards)) + 1):
        need = target_len - wild_len
        if need < 0 or need > len(available):
            continue
        for fixed in combinations(available, need):
            for wild in combinations(wildcards, wild_len):
                combos.append(tuple(list(fixed) + list(wild)))
    return combos


def _build_sequence_combos(cards: Sequence[Card], trump_level: Rank) -> List[Tuple[Card, ...]]:
    sequences = [
        (14, 2, 3, 4, 5),
        (2, 3, 4, 5, 6),
        (3, 4, 5, 6, 7),
        (4, 5, 6, 7, 8),
        (5, 6, 7, 8, 9),
        (6, 7, 8, 9, 10),
        (7, 8, 9, 10, 11),
        (8, 9, 10, 11, 12),
        (9, 10, 11, 12, 13),
        (10, 11, 12, 13, 14),
    ]
    combos: List[Tuple[Card, ...]] = []
    rank_map: Dict[Rank, List[Card]] = {}
    wildcards = [card for card in cards if is_wildcard(card, trump_level)]
    for card in cards:
        if card.rank in (SMALL_JOKER_RANK, BIG_JOKER_RANK):
            continue
        if is_wildcard(card, trump_level):
            continue
        rank_map.setdefault(card.rank, []).append(card)
    for sequence in sequences:
        combos.extend(
            _assemble_sequence(sequence, rank_map, wildcards, require_flush=False, suit=None)
        )
        for suit in PLAIN_SUITS:
            combos.extend(
                _assemble_sequence(
                    sequence, rank_map, wildcards, require_flush=True, suit=suit
                )
            )
    return combos


def _build_consecutive_pairs_combos(cards: Sequence[Card], trump_level: Rank) -> List[Tuple[Card, ...]]:
    sequences = [
        (14, 2, 3),
        (2, 3, 4),
        (3, 4, 5),
        (4, 5, 6),
        (5, 6, 7),
        (6, 7, 8),
        (7, 8, 9),
        (8, 9, 10),
        (9, 10, 11),
        (10, 11, 12),
        (11, 12, 13),
        (12, 13, 14),
    ]
    return _build_group_sequence_combos(cards, trump_level, sequences, group_size=2)


def _build_steel_plate_combos(cards: Sequence[Card], trump_level: Rank) -> List[Tuple[Card, ...]]:
    sequences = [(rank, rank + 1) for rank in range(2, 14)]
    sequences.append((13, 14))
    return _build_group_sequence_combos(cards, trump_level, sequences, group_size=3)


def _build_group_sequence_combos(
    cards: Sequence[Card],
    trump_level: Rank,
    sequences: Sequence[Tuple[Rank, ...]],
    group_size: int,
) -> List[Tuple[Card, ...]]:
    combos: List[Tuple[Card, ...]] = []
    rank_map: Dict[Rank, List[Card]] = {}
    wildcards = [card for card in cards if is_wildcard(card, trump_level)]
    for card in cards:
        if is_wildcard(card, trump_level):
            continue
        if card.rank in (SMALL_JOKER_RANK, BIG_JOKER_RANK):
            continue
        rank_map.setdefault(card.rank, []).append(card)
    for sequence in sequences:
        combos.extend(
            _assemble_group_sequence(sequence, rank_map, tuple(wildcards), group_size)
        )
    return combos


def _assemble_group_sequence(
    sequence: Tuple[Rank, ...],
    rank_map: Dict[Rank, List[Card]],
    wildcards: Tuple[Card, ...],
    group_size: int,
) -> List[Tuple[Card, ...]]:
    combos: List[Tuple[Card, ...]] = []

    def helper(
        index: int,
        used_ids: Tuple[int, ...],
        remaining_wildcards: Tuple[Card, ...],
        chosen: Tuple[Card, ...],
    ) -> None:
        if index == len(sequence):
            combos.append(chosen)
            return
        rank = sequence[index]
        available = tuple(
            card
            for card in rank_map.get(rank, [])
            if card.card_id not in used_ids
        )
        max_wild = min(group_size, len(remaining_wildcards))
        for wild_used in range(max_wild + 1):
            need = group_size - wild_used
            if len(available) < need:
                continue
            for fixed in combinations(available, need):
                if wild_used == 0:
                    helper(
                        index + 1,
                        used_ids + tuple(card.card_id for card in fixed),
                        remaining_wildcards,
                        chosen + fixed,
                    )
                else:
                    for wild_indices in combinations(range(len(remaining_wildcards)), wild_used):
                        selected = tuple(remaining_wildcards[i] for i in wild_indices)
                        next_remaining = tuple(
                            card
                            for idx, card in enumerate(remaining_wildcards)
                            if idx not in wild_indices
                        )
                        helper(
                            index + 1,
                            used_ids + tuple(card.card_id for card in fixed),
                            next_remaining,
                            chosen + fixed + selected,
                        )

    helper(0, tuple(), wildcards, tuple())
    return combos


def _assemble_sequence(
    sequence: Tuple[Rank, ...],
    rank_map: Dict[Rank, List[Card]],
    wildcards: Sequence[Card],
    require_flush: bool,
    suit: Optional[Suit],
) -> List[Tuple[Card, ...]]:
    combos: List[Tuple[Card, ...]] = []

    def backtrack(
        index: int,
        used_ids: Tuple[int, ...],
        remaining_wildcards: Tuple[Card, ...],
        chosen: Tuple[Card, ...],
    ) -> None:
        if index == len(sequence):
            combos.append(chosen)
            return
        rank = sequence[index]
        candidates = [
            card
            for card in rank_map.get(rank, [])
            if card.card_id not in used_ids and (not require_flush or card.suit == suit)
        ]
        for card in candidates:
            backtrack(
                index + 1,
                used_ids + (card.card_id,),
                remaining_wildcards,
                chosen + (card,),
            )
        for idx, wild in enumerate(remaining_wildcards):
            backtrack(
                index + 1,
                used_ids,
                remaining_wildcards[:idx] + remaining_wildcards[idx + 1 :],
                chosen + (wild,),
            )

    backtrack(0, tuple(), tuple(wildcards), tuple())
    return combos


def apply_move(
    player: PlayerID,
    cards: Optional[Sequence[Card]],
    state: HandState,
) -> HandState:
    if player != state.current_player:
        raise ValueError("Not this player's turn")
    if player in state.finished_order:
        raise ValueError("Player already finished")
    trick = state.trick
    hands = [list(hand) for hand in state.hands]
    if cards is None:
        if trick.pattern is None:
            raise ValueError("Leader cannot pass")
        new_passes = list(trick.passes)
        new_passes[player] = True
        next_player = _next_active_player(player, state.finished_order)
        completed = (
            trick.last_player is not None
            and all(
                new_passes[idx]
                for idx in range(4)
                if idx != trick.last_player and idx not in state.finished_order
            )
        )
        if completed and trick.last_player is not None:
            reset_passes = (False, False, False, False)
            next_leader = trick.last_player
            if next_leader in state.finished_order:
                next_leader = _next_active_player(next_leader, state.finished_order)
            return HandState(
                trump_level=state.trump_level,
                hands=tuple(tuple(cards) for cards in hands),
                current_player=next_leader,
                trick=TrickState(
                    leader=next_leader,
                    last_player=None,
                    pattern=None,
                    passes=reset_passes,
                ),
                finished_order=state.finished_order,
            )
        return HandState(
            trump_level=state.trump_level,
            hands=tuple(tuple(cards) for cards in hands),
            current_player=next_player,
            trick=TrickState(
                leader=trick.leader,
                last_player=trick.last_player,
                pattern=trick.pattern,
                passes=tuple(new_passes),
            ),
            finished_order=state.finished_order,
        )
    combo = tuple(cards)
    for card in combo:
        if card not in hands[player]:
            raise ValueError("Card not in hand")
    pattern = classify_pattern(combo, state)
    if pattern is None:
        raise ValueError("Invalid pattern")
    if trick.pattern is not None and not can_beat(pattern, trick.pattern, state):
        raise ValueError("Pattern does not beat current trick")
    for card in combo:
        hands[player].remove(card)
    new_hands = tuple(tuple(sort_cards(hand, state.trump_level)) for hand in hands)
    new_finished = list(state.finished_order)
    if not new_hands[player]:
        new_finished.append(player)
    next_player = _next_active_player(player, tuple(new_finished))
    return HandState(
        trump_level=state.trump_level,
        hands=new_hands,
        current_player=next_player,
        trick=TrickState(
            leader=trick.leader if trick.pattern is not None else player,
            last_player=player,
            pattern=pattern,
            passes=(False, False, False, False),
        ),
        finished_order=tuple(new_finished),
    )


def _next_active_player(current: PlayerID, finished_order: Tuple[PlayerID, ...]) -> PlayerID:
    if len(finished_order) >= 4:
        return current
    next_player = (current + 1) % 4
    while next_player in finished_order:
        next_player = (next_player + 1) % 4
    return next_player


def compute_tribute(
    finished_order: Sequence[PlayerID],
    hands: Sequence[Sequence[Card]],
    trump_level: Rank,
) -> TributePlan:
    if len(finished_order) != 4:
        raise ValueError("Finished order must contain 4 players")
    head = finished_order[0]
    tail = finished_order[-1]
    second = finished_order[1]
    double_down = _team_id(finished_order[-1]) == _team_id(finished_order[-2])
    donors = [finished_order[-1]]
    if double_down:
        donors.append(finished_order[-2])
    anti = _check_anti_tribute(donors, hands)
    if anti:
        return TributePlan(exchanges=tuple(), leader=head, anti_tribute=True)
    exchanges: List[TributeExchange] = []
    if double_down:
        tribute_cards = [
            (donor, _highest_non_wildcard(hands[donor], trump_level)) for donor in donors
        ]
        tribute_cards.sort(key=lambda item: card_sort_key(item[1], trump_level), reverse=True)
        recipients = [head, second]
        for idx, (donor, card) in enumerate(tribute_cards):
            recipient = recipients[min(idx, len(recipients) - 1)]
            refund = _lowest_refund_card(hands[recipient])
            exchanges.append(
                TributeExchange(
                    donor=donor,
                    recipient=recipient,
                    tribute_card=card,
                    refund_card=refund,
                )
            )
        if len(tribute_cards) > 1 and card_sort_key(tribute_cards[0][1], trump_level) == card_sort_key(
            tribute_cards[1][1], trump_level
        ):
            leader = (head + 1) % 4
        else:
            leader = tribute_cards[0][0]
    else:
        donor = donors[0]
        tribute_card = _highest_non_wildcard(hands[donor], trump_level)
        refund = _lowest_refund_card(hands[head])
        exchanges.append(
            TributeExchange(
                donor=donor,
                recipient=head,
                tribute_card=tribute_card,
                refund_card=refund,
            )
        )
        leader = donor
    return TributePlan(exchanges=tuple(exchanges), leader=leader, anti_tribute=False)


def _team_id(player: PlayerID) -> int:
    return 0 if player % 2 == 0 else 1


def _has_double_big_jokers(players: Sequence[PlayerID], hands: Sequence[Sequence[Card]]) -> bool:
    big_jokers = 0
    for player in players:
        big_jokers += sum(1 for card in hands[player] if card.rank == BIG_JOKER_RANK)
    return big_jokers >= 2


def _check_anti_tribute(
    donors: Sequence[PlayerID],
    hands: Sequence[Sequence[Card]],
) -> bool:
    if len(donors) == 1:
        return sum(1 for card in hands[donors[0]] if card.rank == BIG_JOKER_RANK) == 2
    unique_players = set(donors)
    total_big = 0
    for player in unique_players:
        total_big += sum(1 for card in hands[player] if card.rank == BIG_JOKER_RANK)
    return total_big >= 2


def _highest_non_wildcard(cards: Sequence[Card], trump_level: Rank) -> Card:
    eligible = [
        card
        for card in cards
        if not is_wildcard(card, trump_level)
    ]
    if not eligible:
        raise ValueError("No eligible tribute card")
    return max(eligible, key=lambda card: card_sort_key(card, trump_level))


def _lowest_refund_card(cards: Sequence[Card]) -> Optional[Card]:
    eligible = [card for card in cards if card.rank <= 10]
    if not eligible:
        return None
    return min(eligible, key=lambda card: card.rank)


def update_levels_after_hand(
    finished_order: Sequence[PlayerID],
    current_even: Rank,
    current_odd: Rank,
) -> LevelUpdate:
    if not finished_order:
        raise ValueError("Empty finish order")
    winner_team = "even" if _team_id(finished_order[0]) == 0 else "odd"
    winner_team_id = 0 if winner_team == "even" else 1
    losing_team_id = 1 - winner_team_id
    last = finished_order[-1]
    second_last = finished_order[-2]
    if _team_id(last) == losing_team_id and _team_id(second_last) == losing_team_id:
        upgrade = 3
    elif _team_id(last) == losing_team_id:
        upgrade = 2
    else:
        upgrade = 1
    if winner_team == "even":
        new_even = min(14, current_even + upgrade)
        new_odd = current_odd
    else:
        new_odd = min(14, current_odd + upgrade)
        new_even = current_even
    passed_A = _check_pass_A(finished_order, new_even, new_odd, winner_team)
    return LevelUpdate(
        new_even=new_even,
        new_odd=new_odd,
        winner_team=winner_team,
        passed_A=passed_A,
    )


def _check_pass_A(
    finished_order: Sequence[PlayerID],
    level_even: Rank,
    level_odd: Rank,
    winner_team: str,
) -> bool:
    target_level = level_even if winner_team == "even" else level_odd
    if target_level != 14:
        return False
    head = finished_order[0]
    partner = head + 2 if head < 2 else head - 2
    if finished_order[-1] == partner:
        return False
    return True


def parse_cards(tokens: Sequence[str]) -> Tuple[Card, ...]:
    deck_map = {
        "S": 0,
        "H": 1,
        "C": 2,
        "D": 3,
        "J": JOKER_SUIT,
    }
    rank_map = {
        "A": 14,
        "K": 13,
        "Q": 12,
        "J": 11,
        "T": 10,
    }
    cards: List[Card] = []
    for idx, token in enumerate(tokens):
        token = token.strip()
        if not token:
            continue
        upper = token.upper()
        if upper in ("BJ", "SJ"):
            rank = BIG_JOKER_RANK if upper == "BJ" else SMALL_JOKER_RANK
            cards.append(Card(rank=rank, suit=JOKER_SUIT, card_id=idx))
            continue
        suit_symbol = token[0].upper()
        suit = deck_map[suit_symbol]
        rank_text = token[1:].upper()
        if rank_text in rank_map:
            rank = rank_map[rank_text]
        else:
            rank = int(rank_text)
        cards.append(Card(rank=rank, suit=suit, card_id=idx))
    return tuple(cards)

