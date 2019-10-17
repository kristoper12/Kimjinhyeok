import tkinter as tk
import numpy as np
import random
from keras.models     import Sequential
from keras.models     import load_model, model_from_json
from keras.layers     import Dense
from keras.optimizers import SGD

learning_rate = 0.05
reduction = 0.9

def get_max(data) :
    if data[0][0] > data[0][1] :
        return 0
    else :
        return 1

def get_data(x, y, z) :
    data_x = np.array([[0.0 for _ in range(y)] for _ in range(x)])
    data_y = np.array([[0 for _ in range(z)] for _ in range(x)])
    for i in range(x) :
        for j in range(y) :
            data_x[i][j] = random.uniform(0, 1)
        if random.randrange(0, 2) == 0 :
            data_y[i][0] = 0
            data_y[i][1] = 1
        else :
            data_y[i][0] = 1
            data_y[i][1] = 0

    return data_x, data_y

def get_predict(s):
    return model.predict(s)

class Game(tk.Frame) :
    check = 0
    count = 0
    before_bar_position = 0
    t_x = np.array([[0.0 for _ in range(5)] for _ in range(2)])
    t_y = np.array([0.0 for _ in range(2)])
    x_data = np.array([[0.0 for _ in range(5)] for _ in range(1)])
    y_data = np.array([0.0 for _ in range(2)])
    # master mean parent. so, it mean Tk
    def __init__(self, master):
        super(Game, self).__init__(master)
        self.width = 600
        self.height = 400
        self.canvas = tk.Canvas(self, bg = '#396FDA', width = self.width, height = self.height)
        # Declared to be used out of space, and allocates space to be used.
        # In other words, it serves to eliminate unnecessary space.
        self.canvas.pack()
        self.pack()

        self.ball = Ball(self.canvas, self.width / 2, self.height / 2)

        self.items = {}
        self.bar = Bar(self.canvas, 30, 50)
        self.items[self.bar.item] = self.bar
        # self.enemy_bar = Bar(self.canvas, 560, self.height / 2)
        # self.items[self.enemy_bar.item] = self.enemy_bar

        self.game_loop()
        self.canvas.focus_set()
        # self.canvas.bind('<Up>',
        #                  lambda _: self.bar.move(0, -10))
        # self.canvas.bind('<Down>',
        #                  lambda _: self.bar.move(0, 10))

    def game_loop(self):
        temp = np.array([0.0 for _ in range(2)])
        before_x = np.array([[0.0 for _ in range(5)] for _ in range(1)])
        action = 0
        reward = 0
        done = self.check_collisions()
        ball_coords = self.ball.get_position()
        bar_coords = self.bar.get_position()
        # before_bar_position = self.ball.get_position()
        # done = self.ball.update()
        temp = self.y_data
        before_x = self.x_data

        self.after(25, self.game_loop)
        self.x_data[0][0] = ball_coords[0] / 600.0
        self.x_data[0][1] = ball_coords[1] / 400.0
        self.x_data[0][2] = bar_coords[1] / 350.0
        dir = self.ball.direction

        #### x, y
        if dir[0] == 1 :
            self.x_data[0][3] = 0
        elif dir[0] == -1 :
            self.x_data[0][3] = 1
        if dir[1] == 1 :
            self.x_data[0][4] = 0
        elif dir[1] == -1 :
            self.x_data[0][4] = 1
        # if dir[0] == 1 and dir[1] == 1 :
        #     self.x_data[0][3] = 0
        # elif dir[0] == 1 and dir[1] == -1 :
        #     self.x_data[0][3] = 0.33
        # elif dir[0] == -1 and dir[1] == 1 :
        #     self.x_data[0][3] = 0.66
        # elif dir[0] == -1 and dir[1] == -1 :
        #     self.x_data[0][3] = 0.99


        self.y_data = get_predict(self.x_data)
        if get_max(self.y_data) == 0 :
            if (bar_coords[1] > 10) :
                self.bar.move(0, -10)
            else :
                self.bar.move(0, 0)
        else :
            if (bar_coords[1] < 350):
                self.bar.move(0, 10)
            else :
                self.bar.move(0, 0)

        if done == 1 :
            reward = 5
        elif done == 2 :
            reward = -5



        if self.check == 1 :

            # print(self.x_data)
            max = get_max(self.y_data)
            t_max = get_max(temp)
            #self.y_data[0][max] = self.y_data[0][max] + learning_rate \
            #                      * (reward + reduction * self.y_data[0][max] - temp[0][action])
            temp[0][t_max] = temp[0][t_max] + learning_rate \
                                  * (reward + reduction * self.y_data[0][max] - temp[0][t_max])
            # 추가 되는지 확인
            # Epoch
            # print(self.y_data)
            # print(model.predict((self.x_data)))
            if (done == 1 or done == 2):
                self.count = self.count + 1
                model.fit(self.t_x, self.t_y, verbose=0)
                self.check = 0
                print("[", self.count, "] ", self.y_data)
            else:
                self.t_x = np.append(self.t_x, self.x_data, axis=0)
                self.t_y = np.append(self.t_y, self.y_data, axis=0)
        elif self.check != 1 :
            self.t_x = self.x_data
            self.t_y = self.y_data
            self.check = 1




        # print(self.x_data)
        # ball_position = self.ball.get_position()
        # self.enemy_bar.move(0, ball_position[3] - before_bar_position[3])

    def check_collisions(self):
        ball_coords = self.ball.get_position()
        items = self.canvas.find_overlapping(*ball_coords)
        objects = [self.items[x] for x in items if x in self.items]
        done = self.ball.collide(objects)
        if done == 1 :
            self.ball.update()
            return 1
        else :
            return self.ball.update()

class GameObject :
    # the object needs to be drawn on the canvas and needs its identity (item)
    def __init__(self, canvas, item):
        self.canvas = canvas
        self.item = item

    def move(self, x, y):
        # the move method moves an item to the given argument, (x, y)
        coords = self.get_position()
        if coords[1] > 0 and coords[3] < 400 :
            self.canvas.move(self.item, x, y)
        else :
            self.canvas.move(self.item, x, -y)

    def get_position(self):
        return self.canvas.coords(self.item)

class Ball(GameObject) :
    def __init__(self, canvas, x, y):
        self.radius = 10
        self.direction = [1, -1]
        self.speed = 10
        item = canvas.create_oval(x - self.radius, y - self.radius,
                                  x + self.radius, y + self.radius,
                                  fill = '#FA8541')
        super(Ball, self).__init__(canvas, item)

    def collide(self, game_objects):
        done = 0
        coords = self.get_position()
        x = (coords[0] + coords[2]) * 0.5
        # When a crash
        if len(game_objects) >= 1:
            game_object = game_objects[0]
            coords = game_object.get_position()
            if x > coords[2]:
                self.direction[0] = 1
                done = 1
            elif x < coords[0]:
                self.direction[0] = -1
            else:
                self.direction[1] *= -1
        return done

    def update(self):
        coords = self.get_position()
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        done = 0
        if coords[0] <= 0 or coords[2] >= width :
            self.direction[0] *= -1
            if coords[0] <= 0 :
                done = 2
        if coords[1] <= 10 or coords[3] >= height - 14 :
            self.direction[1] *= -1

        x = self.direction[0] * self.speed
        y = self.direction[1] * self.speed

        self.move(x, y)
        return done

class Bar(GameObject):
    def __init__(self, canvas, x, y):
        self.width = 10
        self.height = 40
        item = canvas.create_rectangle(x - self.width / 2, y - self.height / 2,
                                       x + self.width / 2, y + self.height / 2,
                                       fill = '#FA4A41')
        super(Bar, self).__init__(canvas, item)

if __name__ == '__main__' :
    model = Sequential()
    model.add(Dense(128, input_dim=5, activation='relu'))
    model.add(Dense(52, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='mse', optimizer=SGD(learning_rate = 0.01, momentum = 0.9))

    data_x, data_y = get_data(200, 5, 2)
    model.fit(data_x, data_y, verbose=0)

    root = tk.Tk()
    # this decide to game's name
    root.title('JH-Pong')
    game = Game(root)
    # This section refers to the act of blood circulation.
    game.mainloop()