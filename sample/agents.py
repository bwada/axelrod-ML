from dataclasses import dataclass
import copy
import random
import torch

@dataclass
class always_cheat:
    def make_move(self):
        return 0

    def update(self,other_move):
        pass

    def spawn_child(self):
        return always_cheat()
    
    def reset(self):
        pass

@dataclass
class always_colab:
    def make_move(self):
        return 1
    
    def update(self, other_move):
        pass

    def spawn_chile(self):
        return always_colab()

    def reset(self):
        pass

@dataclass 
class tit_for_tat:
    op_prev_move: int

    def __init__(self):
        self.op_prev_move = 1

    def make_move(self):
        if self.op_prev_move == 1:
            return 1
        else:
            return 0

    def update(self, other_move):
        self.op_prev_move = other_move
    
    def spawn_child(self):
        return tit_for_tat()

    def reset(self):
        self.op_prev_move = 1

@dataclass
class grudge_holder:
    betrayed: bool

    def __init__(self):
        self.betrayed = False
    
    def make_move(self):
        if self.betrayed:
            return 0
        else:
            return 1
    
    def update(self, other_move):
        if other_move == 0:
            self.betrayed = True
    
    def spawn_child(self):
        return grudge_holder()
        
    def reset(self):
        self.betrayed = False

@dataclass
class human_player:
    def make_move(self):
        move = None
        while not ((move == 0) or (move == 1)):
            move = int(input("Cooperate? Or BETRAY?!? 0 to cheat and 1 to cooperate\n"))
            if not ((move == 0) or (move == 1)):
                print("Invalid input")
        return move
    
    def update(self, other_move):
        pass

    def spawn_child(self):
        return human_player()
    
    def reset(self):
        pass

@dataclass
class simple_rnn:
    model: torch.nn.RNN
    hidden_size: int
    previous_hidden: torch.tensor
    # are two dimmensional and assume the first value is this agent's own move and the second is the opponent's
    init_move: torch.tensor
    previous_move: torch.tensor
    # self.own_move is defined on the fly to store the users last move
    # If I had more time, I'd think more about how I want to implement this to avoid messing with typing

    mutation_scale = 0.05

    def __init__(self,hidden_size=3):
        self.model = torch.nn.RNN(2,hidden_size)
        for param in self.model.parameters():
            param.requires_grad = False
        self.hidden_size = hidden_size
        self.previous_hidden = torch.zeros(hidden_size)
        self.init_move = torch.rand(2)
        self.previous_move = self.init_move.clone()

    def make_move(self):
        # The input stores 0 as betray and 1 as cooperate
        # We rescale the input so -1 is betray and 1 is cooperate
        # Note that the pytorch default RNN uses tanh, which I use for rescaling
        # The first element of the hidden layer is taken to be the move this actor makes, the rest is arbitrary memory
        input = (2*self.previous_move - 1).reshape((1,2))
        output_hidden = self.model(input, self.previous_hidden.reshape((1,self.hidden_size)))[0]
        self.previous_hidden = output_hidden.clone()
        value = output_hidden.reshape(-1)[0].item()
        value = (value + 1)/2
        #print(f"value: {value}")
        self.own_move = round(value)
        return(self.own_move)

    def update(self, other_move):
        # because the model is small, I don't mind evaluating it twice
        own_move = self.own_move
        self.previous_move = torch.tensor([own_move,other_move]).float()

    def spawn_child(self):
        # because the model is small, I don't mind making an extra and overwriting it
        child = simple_rnn(self.hidden_size)
        child_model = copy.deepcopy(self.model)
        for param in child_model.parameters():
            param += self.mutation_scale * torch.rand_like(param)
        child_init_move = copy.deepcopy(self.init_move)
        child_init_move += torch.rand_like(child_init_move)
        child.model = child_model
        child.init_move = child_init_move
        return child

    def reset(self):
        self.previous_move = self.init_move.clone()
        self.previous_hidden.zero_()
        self.own_move = None

if __name__ == "__main__":
    #some simple tests
    r = simple_rnn(3)
    print(r.make_move())
