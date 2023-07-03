import random
import numpy as np
import matplotlib.pyplot as plt
from utils import Player, Device

from greedy import Greedy, E_Greedy

class Smart(Player):
    def __init__(self, coins: int, CF, eps):
        super().__init__(coins)
        self.CF = CF
        self.eps = eps
        self.K2 = (1.0/(1-self.CF))

    def logic(self) -> Device:
        for dev in self.devices:
            min_n = dev.estimate_variance*self.K2/self.eps**2
            if min_n > .1*(self.coins-len(self.history_choices)):
                continue
            print(min_n)
            if dev.num_play < min_n:
                best_dev = dev
                break
        else:
            best_dev = sorted(self.devices, key=lambda x:x.estimate_mean)[0]
        return best_dev
 
class EconomicallySmart(Player):
    def __init__(self, coins: int, CF, eps):
        super().__init__(coins)
        self.CF = CF
        self.eps = eps
        self.K2 = (1.0/(1-self.CF))

    def logic(self) -> Device:
        for dev in self.devices:
            if dev.estimate_mean < 0:
                continue
            min_n = dev.estimate_variance*self.K2/self.eps**2
            
            print(min_n)
            if dev.num_play < min_n:
                best_dev = dev
                break
        else:
            best_dev = sorted(self.devices, key=lambda x:x.estimate_mean)[0]
        return best_dev

if __name__ == "__main__":
    greedy = Greedy(50)
    e_greedies = []
    for i,eps in zip(range(1),[1,.05,.1,.15,.2]):
        e_greedies.append(E_Greedy(500,eps))
    smart = Smart(50, .75,.2)
    e_smart = EconomicallySmart(500, .75, .2)

    #greedy.play()
    for e_greedy in e_greedies:
        e_greedy.play()
    smart.play()
    e_smart.play()

    fig = plt.figure(figsize=(8, 6))  
    # رسم نمودارها
    plt.plot(e_smart.avg_episodes_everysteps()[4:], label='greedy')
    for e_greedy in e_greedies: 
        plt.plot(e_greedy.avg_episodes_everysteps()[4:], label=f'e_greedy_{e_greedy.eps}')
    plt.plot(smart.avg_episodes_everysteps(), label='smart')

    # افزودن عنوان و برچسب محورها
    plt.title('avg rewards - steps')
    plt.xlabel('avg rewards')
    plt.ylabel('steps')

    # افزودن لیژاند
    plt.legend()

    # نمایش نمودارها
    plt.show()
    stop