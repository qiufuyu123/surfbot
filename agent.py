import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import visualtorch
from torchsummary import summary
import numpy as np
import random
import cv2
import time
import os
from tqdm import trange
class Agent:
    def __init__(self):
        self.use_cuda = torch.cuda.is_available()
        print(self.use_cuda)

        self.model = nn.Sequential(
            
            nn.Conv2d(1,32,(8,8),(4,4)),
            nn.ReLU(),
            nn.Conv2d(32,64,(4,4),(2,2)),
            nn.ReLU(),
            nn.Conv2d(64,64,3,1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,3)
        )
        if self.use_cuda:
            self.model = self.model.cuda()
        if os.path.exists("model_v1.pth"):
            self.model.load_state_dict(torch.load('model_v1.pth',map_location=torch.device('cuda' if self.use_cuda else 'cpu')))
        self.runned_time = 0
        self.trained_cnt = 0
        self.memory = []
        self.criterion = nn.MSELoss()
        if self.use_cuda:
            self.criterion = self.criterion.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        
        dummy_input = torch.randn(1,1,64,64)
        if self.use_cuda:
            dummy_input = dummy_input.cuda()
        # img = visualtorch.layered_view(self.model,(1,1,64,64),one_dim_orientation="x",spacing=40,scale_xy=10,type_ignore=[nn.ReLU,nn.Flatten])
        # plt.axis("off")
        # plt.tight_layout()
        # plt.imshow(img)
        # plt.show()
        print(dummy_input)
        print(dummy_input.shape)
        output = self.model(dummy_input).cpu()
        print(output.shape)
        
        #cv2.namedWindow("aaa")
    def forward(self,x):
        xx = torch.tensor(x)
        if self.use_cuda:
            xx = xx.cuda()
        return self.model.forward(xx)
    
    def act(self, state):
        #if self.trained_cnt<1:
        #    return random.randint(0,2)
        qval = self.forward(state).cpu().detach().numpy()[0]
        # prob = F.softmax(torch.from_numpy(qval),0).detach().numpy()
        # v= np.random.choice(range(2),p=prob)
        v = np.argmax(qval)
        #print(f"v:{qval}")
        return v

    def remember(self,state,nextState,action,reward,done,runned_time):
        self.runned_time = runned_time
        #print(f"mem: {action}")
        self.memory.append({"state":state,"next_state":nextState,"act":action,"reward":reward,"done":done})
        
    def learn(self):
        self.batchSize = 128
        sz =  len(self.memory)
        if sz> 1024*4:
            self.memory = []
            print("trimming memory")
            return 0
        if sz < self.batchSize:
            return 0
        batch = random.sample(self.memory,self.batchSize)
        self.learnBatch(batch)
        self.trained_cnt+=1
        return self.trained_cnt

    def fit(self,xTrain,yTrain,epochs):
        self.model.train(True)

        for epoch in trange(epochs):
            running_loss = 0.0
            
            for i in range(self.batchSize):
                self.optimizer.zero_grad()
                #img = np.reshape(xTrain[i],(64,64)).astype(np.uint8)
                
                xi = torch.tensor(np.reshape(xTrain[i],(1,1,64,64)))
                if self.use_cuda:
                    xi = xi.cuda()
                #print(xi.shape)
                outputs = self.model.forward(xi)
                yi = torch.tensor(np.reshape(yTrain[i],(1,3)))
                if self.use_cuda:
                    yi = yi.cuda()
                loss    = self.criterion(outputs,yi)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                #time.sleep(0.1)
            print("Epoch: "+str(self.trained_cnt)+" , loss: "+str(running_loss/self.batchSize))
        self.model.train(False)
        torch.save(self.model.state_dict(),'model_v1.pth')
    
    def backup(self):
        torch.save(self.model.state_dict(),'model_v1-'+str(self.trained_cnt)+'-backup.pth')

    def learnBatch(self,batch,alpha=0.9):
        actions = [i["act"] for i in batch]
        rewards = [i["reward"] for i in batch]
        
        
        state = np.array([i["state"] for i in batch])
        nextState = np.array([i["next_state"] for i in batch])
        #print(state.shape)
        #print(nextState.shape)
        t1 = torch.tensor(np.reshape(state,(self.batchSize,1,64,64)))
        t2 = torch.tensor(np.reshape(nextState,(self.batchSize,1,64,64)))
        if self.use_cuda:
            t1 = t1.cuda()
            t2 = t2.cuda()
        state_pred = self.model(t1)
        nextState_pred = self.model(t2)
        state_pred = state_pred.cpu().detach().numpy()
        nextState_pred = nextState_pred.cpu().detach().numpy()
        #print(state_pred.shape)
        #print(nextState_pred.shape)
        for i in range(self.batchSize):
            act = actions[i]
            reward = rewards[i]
            nexts = nextState_pred[i,act]
            qval = state_pred[i,act]
            if reward < -5:
                state_pred[i,act] = reward
            else:
                state_pred[i, act] += alpha * (reward + 0.95 * np.max(nexts) - qval)
            #print(f"Reward: {i} is {state_pred[i]} with {act}")
        xTrain = np.reshape(state,(self.batchSize,1,64,64))
        yTrain = state_pred
        print(xTrain.shape)
        print(yTrain.shape)
        self.fit(xTrain,yTrain,4)