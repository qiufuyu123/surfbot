import environment as GameEnv
import threading
import time
import pyautogui
from agent import Agent
from torchsummary import summary
agent = Agent()
summary(agent.model,input_size=(1,64,64))

tui = threading.Thread(target= GameEnv.surf_main_thread)
tui.daemon = True
tui.start()
pyautogui.FAILSAFE = False
train_cnt = 0
def reset_game():
    GameEnv.log.info("Wait 5 secs for refresh...")
    pyautogui.click(x=GameEnv.monitor['top']+5,y=GameEnv.monitor['left']+5)
    pyautogui.press('f5')
    time.sleep(3)
    pyautogui.press('space')
    time.sleep(2)
    GameEnv.log.info("Done")
    GameEnv.game_over = 0

def act_game(act):
    if act == 0:
        pyautogui.press('left')
    elif act == 1:
        pyautogui.press('down')
    else:
        pyautogui.press("right")
    time.sleep(0.01)    
    
while not GameEnv.is_prepared:
    time.sleep(1)
# while True:
#    time.sleep(1)
time.sleep(5)
while True:
    if GameEnv.is_pause:
        GameEnv.log.warn("pause!")
        while GameEnv.is_pause:
            time.sleep(1)
    reset_game()
    
    state,reward,done = GameEnv.step()
    epReward = 0
    epTime = time.time()
    stepCounter = 0
    GameEnv.start_train()
    while not done:
        
        action = agent.act(state)
        act_game(action)
        next_state,reward,done = GameEnv.step()
       
        if stepCounter > 700:
            for _ in range(5):
                agent.remember(state,next_state,action,reward,done,stepCounter)
        else:
            agent.remember(state,next_state,action,reward,done,stepCounter)
        
        if done:
            for _ in range(10):
                agent.remember(state,next_state,action,reward,done,stepCounter)
            break
        dur = time.time() - epTime
        #print(dur)
        if dur > 40:
            # abnormal
            break
        state = next_state
        stepCounter +=1
        epReward+=reward
    
    GameEnv.log.info("Reward: "+str(epReward))
    GameEnv.log.info("Wait for learning...")
    pyautogui.click(x=GameEnv.monitor['top']+5,y=GameEnv.monitor['left']+5)
    pyautogui.press('f5')
    epoch = agent.learn()
    if epoch % 50 == 0:
        agent.backup()
    GameEnv.log.info("Train epoch: "+str(epoch))
    