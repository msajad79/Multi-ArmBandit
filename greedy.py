import random
import numpy as np
import matplotlib.pyplot as plt
from utils import Player, Device


class Greedy(Player):
    def __init__(self, coins: int):
        super().__init__(coins)

    def logic(self) -> Device:
        best_dev = sorted(self.devices, key=lambda x:x.estimate_mean, reverse=True)[0]
        return best_dev
    
class E_Greedy(Player):
    def __init__(self, coins: int, eps):
        self.eps = eps
        super().__init__(coins)

    def logic(self) -> Device:
        p = random.random()
        if self.eps > p:
            self.eps = self.eps * .75
            best_dev = random.choice(self.devices)
        else:
            best_dev = sorted(self.devices, key=lambda x:x.estimate_mean)[0]
        return best_dev

if __name__ == "__main__":
    
    greedy = Greedy(100)
    e_greedy = E_Greedy(100,.01)
    greedy.play()
    e_greedy.play()
    stop