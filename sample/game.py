import numpy as np
import random
from typing import Any
from dataclasses import dataclass

import agents
import payouts

MAX_GAME_ITERATIONS = 100
CONST_GAME = True
CONST_GAME_NUM = 10

@dataclass
class game:
    last_game_prob: float
    payout: np.array
    player_1: Any
    player_2: Any
    p1_score: float = 0
    p2_score: float = 0
    print_rounds = False

    def round(self):
        """
        Queries both players strategies, updates them, and returns a boolean to randomly end the game
        """
        p1_act = self.player_1.make_move()
        p2_act = self.player_2.make_move()
        if self.print_rounds:
            print(f"player 1 move:{p1_act}\nplayer 2 move:{p2_act}\n")
        self.p1_score += self.payout[p1_act,p2_act,0]
        self.p2_score += self.payout[p1_act,p2_act,1]
        self.player_1.update(p2_act)
        self.player_2.update(p1_act)

        if random.random() < self.last_game_prob:
            return True
        else:
            return False
    
    def play_game(self):
        if CONST_GAME == True:
            return self.play_const_num_game()
        for i in range(MAX_GAME_ITERATIONS):
            if self.round():
                if self.print_rounds:
                    print(f"Final Score: {self.p1_score}:{self.p2_score}")
                return
        raise Exception("exceeded maximum number of iterations")
    
    def play_const_num_game(self):
        for i in range(CONST_GAME_NUM):
            self.round()
        if self.print_rounds:
            print(f"Final Score: {self.p1_score}:{self.p2_score}")

    def set_printing(self, print_bool):
        self.print_rounds = print_bool

def test_1():
    last_game_prob = 0.2
    player_1 = agents.simple_rnn()
    player_2 = agents.simple_rnn()
    payout = payouts.standard_payout
    g = game(last_game_prob, payout, player_1, player_2)
    g.set_printing(True)
    g.play_game()


def main():
    last_game_prob = 0.2
    player_1 = agents.human_player()
    player_2 = agents.tit_for_tat()
    payout = payouts.standard_payout
    g = game(last_game_prob, payout, player_1, player_2)
    g.set_printing(True)
    g.play_game()

if __name__ == "__main__":
    #test_1()
    main()