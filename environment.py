import mss
import pygetwindow as gw
import cv2
import numpy as np
import time
import threading
from rich.console import Console
import logging
from rich.logging import RichHandler
console = Console()
running_status = 0
game_over = 0
monitor = {}
start_time = 0
# 设置日志的基本配置，包括日志级别、格式和处理器
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # 你可以根据需要自定义日志格式
    datefmt="[%X]",  # 这里设置时间格式，%X 表示24小时制时间（HH:MM:SS）
    handlers=[RichHandler()]
)

# 获取日志器实例
log = logging.getLogger("rich")

# 记录一条信息日志，其中会自动包含时间戳
log.info("Hello, World!")


def nothing():
    pass

def  mse(img1,img2):
    diff = cv2.absdiff(img1,img2)
    diff_sqr = diff ** 2
    mse = np.mean(diff_sqr)
    return mse

def start_train():
    global running_status
    running_status = True

# 设置帧率（每秒帧数）
frame_rate = 30
# 计算每帧之间的时间间隔（秒）
interval = 1 / frame_rate
is_prepared = False
frame_queue = []
current_state = []
fail_times = 0
is_pause = True
def cmd_thread():
    global is_pause
    while True:
        cmd = console.input("[bold red]>[/bold red]")
        if cmd == 'start':
            console.print("start 5 secs later...")
            time.sleep(5)
            console.print("start!")
            start_train()
        elif cmd == 'pause':
            console.print("Pause!")
            is_pause = not is_pause
        # print("ok")
        
def get_reward():
    global game_over
    if game_over:
        return -15
    else:
        return 1

def get_state():
    global frame_queue
    while len(current_state) == 0:
        time.sleep(0.1)
    return np.reshape(current_state,(1,1,64,64)).astype(np.float32)

def step():
    global game_over
    return (get_state(),get_reward(),game_over)

def surf_main_thread():
    global monitor,current_state
    global game_over,running_status,is_prepared
    t1 = threading.Thread(target=cmd_thread)

    t1.daemon=True
    t1.start()
    # 获取所有打开窗口的标题列表
    all_titles = gw.getAllTitles()
    # 打印所有窗口的标题
    for index, title in enumerate(all_titles):
        print(f"{index + 1}. {title}")

    # 用户输入选择的窗口编号
    choice = int(input("请选择要截屏的窗口编号："))
    cv2.namedWindow('main')
    cv2.createTrackbar('posx', 'main', 361, 800,nothing)
    cv2.createTrackbar('posy', 'main', 62, 600  ,nothing)
    cv2.createTrackbar('w', 'main', 78, 800,nothing)
    cv2.createTrackbar('h', 'main', 25, 600  ,nothing)
    cv2.createTrackbar('top', 'main', 0, 100  ,nothing)
    # cv2.createButton("start",start_train,None,cv2.QT_NEW_BUTTONBAR)
    # 检查用户输入是否在有效范围内
    if 1 <= choice <= len(all_titles):
        # 根据用户选择的编号获取窗口标题
        selected_title = all_titles[choice - 1]
        # 根据窗口标题获取窗口对象
        windows = gw.getWindowsWithTitle(selected_title)
        if windows:
            window = windows[0]  # 假设只有一个匹配的窗口
            desired_width = 800  # 指定的窗口宽度
            desired_height = 600  # 指定的窗口高度
            is_prepared = True
            check_region = []
            old_check_region = []
            while True:
            
                # 获取窗口的位置和大小
                monitor = {
                    "top": window.top+cv2.getTrackbarPos('top', 'main'),
                    "left": window.left,
                    "width": window.width,
                    "height": window.height,
                }
                # 创建mss实例
                with mss.mss() as sct:
                    # 截取指定窗口
                    screenshot = sct.grab(monitor)
                    # 将截取的屏幕转换为NumPy数组
                    frame = np.array(screenshot)
                    # 将NumPy数组从RGB格式转换为BGR格式
                    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    # 调整图像大小到指定的窗口大小
                    frame = cv2.resize(frame, (desired_width, desired_height))
                    b,g,r = cv2.split(frame[:,:,:3])
                    b = b[90:desired_height,0:desired_width]
                    b = cv2.resize(b,(64,64))
                    #print(b.shape)
                    pos_x  = cv2.getTrackbarPos('posx', 'main');
                    pos_y  = cv2.getTrackbarPos('posy', 'main');
                    w  = cv2.getTrackbarPos('w', 'main');
                    h  = cv2.getTrackbarPos('h', 'main');
                    check_region = frame[pos_y:pos_y+h,pos_x:pos_x+w]
                    check_region =  cv2.cvtColor(check_region, cv2.COLOR_BGR2GRAY)
                    #ret,check_region = cv2.threshold(check_region,100,255,cv2.THRESH_BINARY)
                    #b[0:32,0:64]=0
                    for y in range(64):
                        for x in range(64):
                            if b[y,x] > 205:
                                b[y,x] = 255
                    #b=cv2.equalizeHist(b)
                    #ret,b = cv2.threshold(b,thres,255,cv2.THRESH_BINARY)
                    frame_queue.append(b)
                    if len(frame_queue) > 4:
                        frame_queue.pop(0)
                        adds = frame_queue[0]//8 + frame_queue[1]//8 + frame_queue[2]//4 + frame_queue[3]//2
                        #adds = np.interp(adds,(adds.min(),adds.max()),(0,1))
                        #print(adds)
                        b = adds
                        current_state = adds
                        if len(old_check_region):
                            diff = mse(check_region,old_check_region)
                            #diff = mse(frame_queue[0],frame_queue[1])
                            #print(diff)
                            if abs(diff) < 0.01 and running_status and not game_over:
                                # if fail_times > 1:
                                    log.warn('Game over!')
                                    game_over=1
                            # else:0
                                # fail_times+=1
                        # else:
                            # fail_times = 
                    old_check_region = check_region
                    check_region2 = cv2.resize(check_region,(800,200))
                    b = cv2.resize(b,(800,600))
                    # 使用OpenCV显示截取的图片
                    b = np.vstack((b,check_region2))

                    cv2.imshow('main', b)
                    # cv2.imshow('main2',r)
                    # 按 'q' 退出
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    
                    # 等待下一帧的时间间隔
                    time.sleep(interval)
        else:
            print("没有找到匹配的窗口。")
    else:
        print("选择的编号超出范围。")

    # 释放所有资源
    cv2.destroyAllWindows()