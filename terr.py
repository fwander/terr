
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import numpy as np
import pickle
import copy

pygame.init()
WIDTH = 800
HIGHT = 600
SIDE_BAR = 100
screen = pygame.display.set_mode((WIDTH + SIDE_BAR, HIGHT))

done = False
min_to_split = 20

clock = pygame.time.Clock()


best = {}
best_colors = []
best_max_size = 10

def draw_best():
    to_draw = list(best.values())
    for i in range(len(to_draw)):
        pygame.draw.circle(screen,to_draw[i].color, (WIDTH + int(SIDE_BAR/2), i * 40 + 15),to_draw[i].r)


def add_to_best(animal):

    global best
    for color in best_colors:
        if animal.color == color:
            return
    if len(best) < best_max_size:
        best[animal.life_time] = animal.copy()
        best_colors.append(animal.color)

        return
    if animal.life_time > min(best.keys()):
        best_colors.remove(best[min(best.keys())].color)
        best.pop(min(best.keys()))
        best[animal.life_time] = animal.copy()
        best_colors.append(animal.color)


class obj:
    def __init__(self,color,size):
        self.r = size
        self.color = color
        self.location = np.array([random.randrange(0,WIDTH),random.randrange(0,HIGHT)])

    def get_at(self, location):
        for o in objs:
            x = o.location[0]
            y = o.location[1]
            dist = (location[0]-x)*(location[0]-x) + (location[1]-y)*(location[1]-y)
            if dist <= o.r * o.r + self.r * self.r:
                return o
    def update(self):
        pass

    def draw(self):
        pass

class animal(obj):
    def __init__(self, color, size, num_eyes=0):
        super().__init__(color,size)
        self.vel = random.randrange(0,360)
        self.energy = 7
        self.speed = 2/self.r+.5
        if num_eyes == 0:
            self.num_eyes = random.randrange(1,6)
        else:
            self.num_eyes = num_eyes
        self.eye_spacing = random.randrange(20,120)/100
        self.life_time = 0
        self.dist = random.randrange(30,120)
        self.brain = model = torch.nn.Sequential(
        torch.nn.Linear(self.num_eyes*2, 40),
        torch.nn.ReLU(),
        torch.nn.Linear(40, 1),
        )

    def copy(self):
        other = animal(self.color,self.r,self.num_eyes)
        other.brain = self.brain

        other.eye_spacing = self.eye_spacing
        other.dist = self.dist
        return other

    def split(self):

        self.energy = min_to_split/2
        other = animal(self.color,self.r,self.num_eyes)

        other.eye_spacing = self.eye_spacing
        other.energy = 7
        other.dist = self.dist
        other.brain = copy.deepcopy(self.brain)
        with torch.no_grad():
            for name,layer in other.brain.named_parameters():
                if random.randrange(20) == 0:
                    layer += layer.clone().fill_(float(random.randrange(-100,100)/1000))

        objs.append(other)

    def get_inputs(self):
        inputs = []

        l = torch.tensor([item for sublist in self.march(self.dist) for item in sublist])
        #print(l)
        return l

    def march(self,distance):
        lines = []
        points = []
        for ray in range(self.num_eyes):
            dir = self.vel-(self.eye_spacing*int((self.num_eyes/2)))/2 + ray*self.eye_spacing
            x = self.location[0] + math.cos(dir) * distance
            y = self.location[1] + math.sin(dir) * distance
            line = np.array([x,y])-self.location
            lines.append(line)
            points.append([float(0.0),float(1.0)])
        P1 = self.location

        for o in objs:
            if o is self:
                continue
            Q = o.location
            r = o.r
            c = P1.dot(P1) + Q.dot(Q) - 2 * P1.dot(Q) - r**2
            for i in range(self.num_eyes):
                V = lines[i]

                a = V.dot(V)
                b = 2 * V.dot(P1 - Q)
                disc = b**2 - 4 * a * c
                if disc < 0:

                    continue
                sqrt_disc = math.sqrt(disc)
                t1 = (-b + sqrt_disc) / (2 * a)
                t2 = (-b - sqrt_disc) / (2 * a)
                if not (0 <= t1 <= 1 or 0 <= t2 <= 1):

                    continue
                points[i] = [float(o.r/self.r)-1,float(np.linalg.norm(o.location-P1)/distance)]
                pygame.draw.line(screen, (200,50,50),P1,P1+V)
        return points

    def update(self):
        self.life_time += 1
        if self.energy >= min_to_split:
            self.split()

        if self.energy <= 0:
            add_to_best(self)
            objs.remove(self)
            return
        self.vel += self.brain(self.get_inputs()).item()

        self.energy -= self.r/500
        dx = self.location[0] + math.cos(self.vel) * self.speed
        dy = self.location[1] + math.sin(self.vel) * self.speed

        self.location = wrap(np.asarray([dx,dy]))
        other = self.get_at(self.location)
        if other is not None and other is not self and other.color != self.color:
            if other.r < self.r:
                self.on_eat(other)
            else:
                other.on_eat(self)

    def on_eat(self, other):
        global foodC
        self.energy += other.r
        if isinstance(other,food):
            foodC -= 1
        else:
            add_to_best(other)
        objs.remove(other)
    def draw(self):
        pygame.draw.circle(screen, self.color, tuple(map(int,self.location)), self.r)
        pygame.draw.line(screen,(0,0,0),self.location, (int(self.location[0]+math.cos(self.vel)*self.r),int(self.location[1]+math.sin(self.vel)*self.r)))

class food(obj):
    def update(self):
        other = self.get_at(self.location)
        if other is not None and other is not self and not isinstance(other,food):
            other.on_eat(self)
    def draw(self):
        pygame.draw.circle(screen, self.color, tuple(map(int,self.location)), self.r)

def wrap(location):
    if location[0] < 0:
        return np.array([WIDTH,location[1]])
    if location[0] > WIDTH:
        return np.array([0,location[1]])
    if location[1] < 0:
        return np.array([location[0], HIGHT])
    if location[1] > HIGHT:
        return np.array([location[0], 0])
    return location

objs = []


def fill_animals(num):
    i = 0
    for i in range(min(int(num/2),len(best.values()))):
        objs.append(list(best.values())[(i+random.randrange(0,10))%len(best.values())])

    for i in range(num):
        objs.append(animal((random.randrange(50,250), random.randrange(50,250),random.randrange(50,250)),random.randrange(5,12)))

def fill_food(num):
    for i in range(num):
        objs.append(food((10,230,40),4))

def all(list):
    for a in list:
        a.__class__.update(a)
        a.__class__.draw(a)


fill_food(50)
fill_animals(15)
timestamp = 50
time = 0
foodC = 50
max_a = 15
max_food = 50
Line_Thicc = 15
while not done:
        screen.fill((0, 0, 0))
        all(objs)
        pygame.draw.line(screen,(40,40,40),(WIDTH+int(Line_Thicc/2),0),(WIDTH+int(Line_Thicc/2),HIGHT),Line_Thicc)
        draw_best()
        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                        done = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        objs = []
                        fill_food(50)
                    if event.key == pygame.K_s:
                        with open('best.pickle', 'wb') as handle:
                            pickle.dump(best, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    if event.key == pygame.K_l:
                        with open('best.pickle', 'rb') as handle:
                            best = pickle.load(handle)
                            for a in best.values():
                                best_colors.append(a.color)

        #clock.tick(60)
        time += 1

        if time == timestamp:
            time = 0

            fill_food(max(0,max_food-foodC))
            foodC += max(0,max_food-foodC)
            numA = len(objs) - foodC
            print(best.keys())
            fill_animals(max(0,max_a-numA))
        pygame.display.flip()
