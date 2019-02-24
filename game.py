import pygame, sys
import numpy as np

dx = [1,  0, -1, 0]
dy = [0, -1,  0, 1]

wrongDir = {0: 2, 1: 3, 2: 0, 3: 1}

class Snake:
    def __init__(self, x = 0, y = 0, blockSize = 20):
        self.position  = [[x, y]]
        self.tail      = [[x, y]]
        self.blockSize = blockSize
        self.direction = 0

    def move(self):
        self.tail = self.position[-1].copy()
        for ind in np.arange(len(self.position) - 1, 0, -1):
            self.position[ind] = self.position[ind - 1].copy()

        if (self.direction == 0):
            self.position[0][0] += self.blockSize
        if (self.direction == 1):
            self.position[0][1] += self.blockSize
        if (self.direction == 2):
            self.position[0][0] -= self.blockSize
        if (self.direction == 3):
            self.position[0][1] -= self.blockSize

class Game:

    def __init__(self, width = 1280, height = 720, speed = 1, snakeColor = (0, 255, 0), foodColor = (255, 0, 0)):
        self.score        = 0
        self.gameOver     = False
        self.screenHeight = height
        self.screenWidth  = width
        self.speed        = speed
        self.screenSize   = width, height
        self.snakeColor   = snakeColor
        self.foodColor    = foodColor
        self.player       = Snake()
        self.clock        = pygame.time.Clock()
        self.screen       = pygame.display.set_mode(self.screenSize)
        self.newFood()

    def newFood(self):
        blockSize = self.player.blockSize
        height    = self.screenHeight
        width     = self.screenWidth
        self.foodPosition = [blockSize * np.random.randint(0, width / blockSize), blockSize * np.random.randint(0, height / blockSize)]

    def checkPosition(self):
        headPosition = self.player.position[0]
        height       = self.screenHeight
        width        = self.screenWidth
        if headPosition[0] < 0 or headPosition[0] > width:
            self.gameOver = True
        if headPosition[1] < 0 or headPosition[1] > height:
            self.gameOver = True
        if len(np.unique(self.player.position, axis = 0)) != len(self.player.position):
            self.gameOver = True
        if headPosition == self.foodPosition:
            self.score += 1
            self.player.position.append(self.player.tail.copy())
            self.newFood()
            return True
        return False

    def render(self):
        self.screen.fill((255, 255, 255))
        for i in self.player.position:
            pygame.draw.rect(self.screen, self.snakeColor, [i[0], i[1], self.player.blockSize, self.player.blockSize])
        pygame.draw.rect(self.screen, self.foodColor, [self.foodPosition[0], self.foodPosition[1], self.player.blockSize, self.player.blockSize])
        pygame.display.update()

    def play(self):
        startTicks = pygame.time.get_ticks()
        cur = 1
        maxScore = 0
        numModels = 5
        bots = []
        for i in range(numModels):
            x = Model()
            bots.append(x)
        while True:
            mdlScore = []
            for mdl in bots:
                self.prevDist = -1
                self.player.__init__()
                print("Iteration " + str(cur) + "\nMax score: " + str(maxScore))

                while not self.gameOver:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()

                    # keys = pygame.key.get_pressed()
                    # if (keys[pygame.K_RIGHT]):
                    #     self.player.direction = 0
                    # if (keys[pygame.K_DOWN]):
                    #     self.player.direction = 1
                    # if (keys[pygame.K_LEFT]):
                    #     self.player.direction = 2
                    # if (keys[pygame.K_UP]):
                    #     self.player.direction = 3

                    if ((pygame.time.get_ticks() - startTicks) > 1):
                        pred = mdl.predict(self.getState())
                        if (wrongDir[self.player.direction] == pred):
                            self.gameOver = True
                        else:
                            self.player.direction = pred
                            self.player.move()
                            mdl.update(self.getReward(), self.getState())
                        startTicks = pygame.time.get_ticks()
                    if (self.checkPosition()):
                        print(self.score)
                        if (self.score > maxScore):
                            maxScore = self.score
                        mdl.update(10, self.getState())
                        self.prevDist = -1
                    self.render()
                mdl.update(-10, self.getState())
                self.gameOver = False
                mdlScore.append(self.score)
                self.score = 0

                print('Game over \\_-^-_/')
                cur += 1
            x = [x for _, x in sorted(zip(mdlScore, bots), key = lambda x: x[0])][:5].copy()
            new_bots = []
            for i in bots:
                for j in bots:
                    new_bots.append(i.join(j))
            bots = new_bots.copy()

    def getState(self):
        x = self.foodPosition
        y = self.player.position[0]
        z = self.player.position
        size = self.player.blockSize
        print((x[0] <= y[0], x[1] >= y[1], x[0] == y[0], x[1] == y[1], self.player.direction, [x[0] - size, x[1]] in z, [x[0] + size, x[1]] in z, [x[0], x[1] - size] in z, [x[0], x[1] + size] in z))
        return (x[0] <= y[0], x[1] >= y[1], x[0] == y[0], x[1] == y[1], self.player.direction, [x[0] - size, x[1]] in z, [x[0] + size, x[1]] in z, [x[0], x[1] - size] in z, [x[0], x[1] + size] in z)

    def getReward(self):
        x = self.foodPosition
        y = self.player.position[0]
        curDist = abs(x[0] - y[0]) + abs(x[1] - y[1]) + 1
        if self.prevDist == -1:
            self.prevDist = curDist

        if self.prevDist < curDist:
            rew = -1
        else:
            rew = 1

        self.prevDist = curDist
        # print(rew)
        return rew

class Model:
    Q = {}
    LF = 0.25
    DF = 0.6

    def predict(self, state):
        if (state not in self.Q.keys()):
            self.Q[state] = np.random.rand(4)
        self.prevState = state
        self.action = np.argmax(self.Q[state])
        # print(state, self.Q[state])
        return np.argmax(self.Q[state])

    def update(self, reward, state):
        if (state not in self.Q.keys()):
            self.Q[state] = np.random.rand(4)
        # self.Q[self.prevState][self.action] += self.LF * (reward + self.DF * np.max(self.Q[state]) - self.Q[self.prevState][self.action])
        self.Q[self.prevState][self.action] += self.LF * reward

    def join(self, mdl):
        res = Model()
        for key in mdl.Q.keys():
            if (key in self.Q.keys()):
                res.Q[key] = (self.Q[key] + mdl.Q[key]) / 2
            else:
                res.Q[key] = mdl.Q[key]
        return res
