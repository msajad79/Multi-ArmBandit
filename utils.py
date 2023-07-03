import random
import statistics

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Device():
    ID = 0
    def __init__(self, id, mean:float, sigma:float, seed=None) -> None:
        self.id = Device.ID
        Device.ID += 1
        self.mean = mean
        self.sigma = sigma
        self.random_generator = random.Random(seed)
        self.h = []

        self.estimate_mean = 0
        self.estimate_variance = 1
        self.num_play = 0
        
    def play(self, player):
        reward = self.random_generator.gauss(self.mean, self.sigma)

        #update history reward player
        player.history_rewards.append(reward)
        self.h.append(reward)
        #update history choices
        player.history_choices.append(self.id)
        #update mean
        self.mean = (self.mean*self.num_play + reward)/(self.num_play+1)
        #update variance
        if len(self.h) > 30:
            self.estimate_variance = statistics.variance(self.h)
        
        self.num_play += 1

        return reward
    
    
    def y_distribution(self, x):
        return (1 / (self.sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - self.mean) ** 2 / (2 * self.sigma ** 2))
        

class Player():
    info_devices = {}
    def __init__(self, coins:int):
        self.coins :int = coins
        self.n_device :int = 10
        self.n_episode :int = 20
        self.devices :list[Device] = []

        self.backup_episodes = []

        self.history_rewards = []
        self.history_choices = []
        if len(Player.info_devices) == 0:
            self.create_info_devices()
        self.define_devices()

    def create_info_devices(self):
        rg_device = random.Random()
        for e in range(self.n_episode+1):
            Player.info_devices[e] = list()
            for d in range(self.n_device):
                mean = rg_device.gauss(1, 1)
                sigma = abs(rg_device.gauss(1, 1))
                Player.info_devices[e].append((mean,sigma))

    def define_devices(self):
        """
        define n device whith:
        mean -> gauss distibution mean 1 , sigma 1
        sigma -> gauss distibution mean 1 , sigma 1
        """
        self.devices = []
        for i in range(self.n_device):
            target_e = len(self.backup_episodes)
            mean, sigma = Player.info_devices[target_e][i]
            self.devices.append(Device(i, mean, sigma))
        return
    
    def play(self):
        for episode in tqdm(range(self.n_episode)):
            for coin in range(self.coins):
                e = self.logic()
                e.play(self)
            self.backup_episode()
            self.history_choices = []
            self.history_rewards = []
            self.define_devices()


    def logic(self) -> Device:
        return self.devices[0]
    
    def backup_episode(self):
        self.backup_episodes.append({
            "devices": self.devices.copy(),
            "history_choices": self.history_choices.copy(),
            "history_rewards": self.history_rewards.copy()
        })
        
    
    def avg_episodes_everysteps(self):
        """return avg all rewards episodes on  every steps"""
        res = []
        for step in range(self.coins):
            av_e = 0
            for e in self.backup_episodes:
                av_e += sum(e["history_rewards"][:step+1])/(step+1)
            res.append(av_e/len(self.backup_episodes))
        return res

    def plot_devices_distribution(self, episode):
        min_plot = None
        max_plot = None
        for device in self.backup_episodes[episode]['devices']:
            if min_plot is None or (device.mean - 3*device.sigma) < min_plot:
                min_plot = device.mean - 3*device.sigma
            if max_plot is None or (device.mean + 3*device.sigma) > max_plot:
                max_plot = device.mean + 3*device.sigma
        x = np.linspace(min_plot, max_plot, 1000)
        
        fig = plt.figure(figsize=(15, 6))
        for device in self.devices:
            y = device.y_distribution(x)
            plt.plot(x, y, label=f'Distribution Device {device.id}')
        plt.xlabel('X')
        plt.ylabel('Probability Density')
        plt.title('Multiple Distributions')
        plt.legend()
        plt.show()

    def plot_history_choices(self, episode):
        unique_elements, counts = np.unique(self.backup_episodes[episode]["history_choices"], return_counts=True)
        plt.bar(unique_elements, counts)
        plt.xlabel('device')
        plt.ylabel('Count play')
        plt.title('count play every device')
        plt.show()

    
