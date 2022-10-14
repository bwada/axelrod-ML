from dataclasses import dataclass
import numpy as np

import agents
import game
import payouts

@dataclass
class round_robin:
    last_game_prob: float
    payout: np.array
    players: list
    scores: np.array

    def __init__(self, last_game_prob, payout, players):
        self.last_game_prob = last_game_prob
        self.payout = payout
        self.players = players
        self.scores = np.zeros_like(players)

    def round(self):
        for i, player_1 in enumerate(self.players):
            for j, player_2 in enumerate(self.players[:i]):
                g = game.game(self.last_game_prob, self.payout, player_1, player_2)
                g.play_game()
                player_1.reset()
                player_2.reset()
                self.scores[i] += g.p1_score
                self.scores[j] += g.p2_score

    def remove_losers(self, n):
        # tiebreak behavior is dictated by argpartition
        if n == 0:
            return
        loser_indices = self.scores.argpartition(n - 1)[:n]
        loser_indices.sort()
        scores_list = list(self.scores)
        for ind in np.flip(loser_indices):
            self.players.pop(ind)
            scores_list.pop(ind)
        self.scores = np.array(scores_list)

    def propagate_best_winners(self, n):
        # propagate the highest scoring winners
        num_players = len(self.scores)
        winners_indices = self.scores.argpartition(num_players - n - 1)[num_players - n:]
        for ind in winners_indices:
            new_player = self.players[ind].spawn_child()
            self.players.append(new_player)
        self.scores = np.concatenate((self.scores, np.zeros(n)))
    
    def propagate_random_winners(self, n):
        # propagate 
        pass

    def reset(self):
        self.scores = np.zeros_like(self.scores)

def test_tourney_1():
    partners = [agents.always_colab() for i in range(5)]
    traitors = [agents.always_cheat() for i in range(5)]
    players = partners + traitors
    tourney = round_robin(0.2, payouts.standard_payout, players)
    for _ in range(5):
        tourney.round()
        print(tourney.scores)
        tourney.remove_losers(1)
        tourney.propagate_best_winners(1)
        tourney.reset()

def test_tourney_2():
    players = [agents.simple_rnn() for i in range(10)]
    tourney = round_robin(0.2, payouts.standard_payout, players)
    for _ in range(40):
        tourney.round()
        print(tourney.scores)
        tourney.remove_losers(1)
        tourney.propagate_best_winners(1)
        tourney.reset()

def main():
    pass

if __name__ == "__main__":
    #test_tourney_1()
    test_tourney_2()
    main()