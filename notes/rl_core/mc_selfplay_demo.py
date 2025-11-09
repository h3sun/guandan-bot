# rl_core/mc_selfplay_demo.py
# Minimal Monte Carlo self-play demo for a tiny card game.
# Usage:
#   python rl_core/mc_selfplay_demo.py --episodes 10000 --env threecard --eps 0.2 --seed 0

import argparse
import random
from collections import defaultdict
from typing import List, Tuple, Any
import numpy as np


# =========================
# Environments
# =========================

class OneCardEnv:
    """
    Each player gets exactly ONE card (1..13). Action is constrained to 'play your own card'.
    Larger card wins: winner +1, loser -1, tie 0. Single-step episode.
    State is a tuple: (p0_card, p1_card).
    """
    def __init__(self):
        self.hands = None
        self.done = False

    def reset(self) -> Tuple[int, int]:
        deck = list(range(1, 14))
        random.shuffle(deck)
        self.hands = (deck.pop(), deck.pop())
        self.done = False
        return self.hands

    def current_player(self) -> int:
        # Simultaneous-move single step; not used but kept for API symmetry
        return 0

    def legal_actions(self, state: Tuple[int, int], pid: int) -> List[int]:
        # Only legal action is to "play your own card"
        # We'll represent the action as the actual card value for clarity.
        return [state[pid]]

    def step(self, actions: Tuple[int, int]) -> Tuple[Any, List[int], bool, dict]:
        assert not self.done
        a0, a1 = actions
        r = 1 if a0 > a1 else (-1 if a0 < a1 else 0)
        self.done = True
        # next_state is None because episode ends immediately
        return None, [r, -r], True, {}

    def episode_metrics(self):
        return {}


class ThreeCardEnv:
    """
    Each player receives THREE cards (sorted ascending). Each chooses one index (0/1/2) to play.
    Larger played card wins: +1 / -1 / 0. Single-step episode.
    State is a pair of tuples: ((p0_c0,p0_c1,p0_c2), (p1_c0,p1_c1,p1_c2))
    Actions are indices in {0,1,2}. We reveal only legal indices.
    """
    def __init__(self):
        self.hands = None
        self.done = False

    def reset(self) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        deck = list(range(1, 14)) * 1  # single deck is enough here
        random.shuffle(deck)
        p0 = sorted([deck.pop(), deck.pop(), deck.pop()])
        p1 = sorted([deck.pop(), deck.pop(), deck.pop()])
        self.hands = (tuple(p0), tuple(p1))
        self.done = False
        return self.hands

    def current_player(self) -> int:
        # Simultaneous-move single step; not used but kept for API symmetry
        return 0

    def legal_actions(self, state, pid: int) -> List[int]:
        # Choose an index from your own hand
        return [0, 1, 2]

    def step(self, actions: Tuple[int, int]) -> Tuple[Any, List[int], bool, dict]:
        assert not self.done
        a0, a1 = actions
        v0 = self.hands[0][a0]
        v1 = self.hands[1][a1]
        r = 1 if v0 > v1 else (-1 if v0 < v1 else 0)
        self.done = True
        return None, [r, -r], True, {}

    def episode_metrics(self):
        return {}


# =========================
# Monte Carlo Agent (on-policy, every-visit)
# =========================

class MCAgent:
    """
    Simple tabular Monte Carlo agent.
    - Uses a user-supplied key_fn(state, pid) -> hashable state key
    - Uses env.legal_actions(state, pid) to mask actions
    - epsilon-greedy over Q[state_key][action_id]
    """
    def __init__(self, eps: float, key_fn, action_space_size: int):
        self.Q = defaultdict(lambda: np.zeros(action_space_size, dtype=np.float64))
        self.N = defaultdict(lambda: np.zeros(action_space_size, dtype=np.float64))
        self.eps = eps
        self.key_fn = key_fn
        self.action_space_size = action_space_size

    def act(self, state, pid: int, legal_actions: List[int]) -> int:
        """
        Returns an action in the same 'id space' as provided by legal_actions:
        - For OneCardEnv we choose from a singleton [actual_card_value], so we return that value.
        - For ThreeCardEnv we choose among indices [0,1,2], so we return one of them.
        """
        key = self.key_fn(state, pid)
        # Build a local mask aligned to the agent's Q-vector indices
        # We assume actions are in [0..A-1] for indexed env (threecard)
        # or arbitrary integers (onecard). For onecard, we map the SINGLE legal action to index 0.
        if len(legal_actions) == self.action_space_size and set(legal_actions) == set(range(self.action_space_size)):
            # Indexed action space (e.g., [0,1,2]) — direct mapping
            if random.random() < self.eps:
                return random.choice(legal_actions)
            q_row = self.Q[key]
            # break ties by random argmax
            best = np.flatnonzero(q_row == q_row.max())
            return int(np.random.choice(best))
        else:
            # Non-indexed / singleton action space (e.g., [actual_card])
            # There is only one legal action; exploration doesn't matter.
            return legal_actions[0]

    def learn(self, traj: List[Tuple[Any, int, float, int]]):
        """
        traj: list of (state, action, reward, pid) for THIS agent.
        Single-step episodes are fine; MC every-visit update:
          Q[s,a] <- Q[s,a] + (G - Q[s,a]) / N[s,a]
        """
        G = 0.0
        for (s, a, r, pid) in reversed(traj):
            G = r + G
            key = self.key_fn(s, pid)

            # Map action to index in Q-row:
            # If action is already in [0..A-1], use directly; else map singleton to idx 0.
            if isinstance(a, int) and 0 <= a < self.action_space_size:
                a_idx = a
            else:
                a_idx = 0  # singleton case

            self.N[key][a_idx] += 1.0
            self.Q[key][a_idx] += (G - self.Q[key][a_idx]) / self.N[key][a_idx]


# =========================
# Self-play driver
# =========================

def self_play(env_name: str = "threecard",
              episodes: int = 5000,
              eps: float = 0.2,
              seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)

    # Select env & state-key function & action space size
    if env_name.lower() == "onecard":
        env = OneCardEnv()
        # State key for player i is simply (pid, own_card)
        def key_fn(state, pid):
            return ("one", pid, state[pid])
        action_space_size = 1  # singleton action per state
    elif env_name.lower() == "threecard":
        env = ThreeCardEnv()
        # Key is (pid, own_hand_tuple)
        def key_fn(state, pid):
            return ("three", pid, state[pid])
        action_space_size = 3  # indices 0/1/2
    else:
        raise ValueError("Unknown env, choose from {onecard, threecard}")

    agents = [MCAgent(eps=eps, key_fn=key_fn, action_space_size=action_space_size),
              MCAgent(eps=eps, key_fn=key_fn, action_space_size=action_space_size)]

    results = []  # rewards for player 0 per episode

    for _ in range(episodes):
        s = env.reset()

        # Simultaneous single-step actions
        a0 = agents[0].act(s, pid=0, legal_actions=env.legal_actions(s, 0))
        a1 = agents[1].act(s, pid=1, legal_actions=env.legal_actions(s, 1))

        _, rewards, done, _ = env.step((a0, a1))
        assert done

        # Store single-step traj for MC update (still valid MC)
        traj0 = [(s, a0, rewards[0], 0)]
        traj1 = [(s, a1, rewards[1], 1)]
        agents[0].learn(traj0)
        agents[1].learn(traj1)

        results.append(rewards[0])

    # Metrics
    total = len(results)
    wins = sum(1 for r in results if r > 0)
    losses = sum(1 for r in results if r < 0)
    draws = total - wins - losses
    avg_return = float(np.mean(results))

    print(f"[env={env_name}] episodes={episodes} eps={eps} seed={seed}")
    print(f"Player0  平均回报: {avg_return:.4f}")
    print(f"Player0  胜率(含平局): {wins/total:.4f}")
    if (wins + losses) > 0:
        print(f"Player0  胜率(除平局): {wins/(wins+losses):.4f}")
    else:
        print("Player0  胜率(除平局): n/a (all draws)")
    print(f"统计: 胜{wins} 负{losses} 平{draws}")


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5000, help="Number of self-play episodes")
    parser.add_argument("--env", type=str, default="threecard", choices=["onecard", "threecard"])
    parser.add_argument("--eps", type=float, default=0.2, help="Epsilon for epsilon-greedy")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    self_play(env_name=args.env, episodes=args.episodes, eps=args.eps, seed=args.seed)


if __name__ == "__main__":
    main()
