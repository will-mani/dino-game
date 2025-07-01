import cv2
import numpy as np
from PIL import ImageGrab
import time
import pyautogui
from dino import Dino
from obstacle import Obstacle
import threading

def update_obstacle_list(dino: Dino, obstacle_list, game_thresh):
    view = dino.obstacle_view_thresh.copy()
    new_obstacle_list = []
    for i in range(len(obstacle_list)):
        obstacle = obstacle_list[i]
        obstacle.update_position_properties(game_thresh, view, dino.width)
        if obstacle.is_on_screen:
            obstacle.update_pixels_per_sec()
            view[:, :obstacle.end_x + 1] = 127
            new_obstacle_list.append(obstacle)
    
    while True:
        new_obstacle = Obstacle()
        new_obstacle.update_position_properties(game_thresh, view, dino.width)
        if new_obstacle.is_on_screen:
            new_obstacle.update_pixels_per_sec()
            view[:, :new_obstacle.end_x + 1] = 127
            new_obstacle_list.append(new_obstacle)
        else:
            break

    return new_obstacle_list


def capture_screen(image_grab_bbox):
    screenshot = ImageGrab.grab(bbox=image_grab_bbox)
    bgr_screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    resized_screenshot = cv2.resize(bgr_screenshot, (0, 0), fx = 1, fy = 1)
    return resized_screenshot


def revert_colors(thresh_img):
    black_pixels = np.where(thresh_img == 0)[0], np.where(thresh_img == 0)[1]
    white_pixels = np.where(thresh_img == 255)[0], np.where(thresh_img == 255)[1]
    thresh_img[black_pixels] = 255
    thresh_img[white_pixels] = 0
    return thresh_img

def press_up():
    pyautogui.press('up')
    pyautogui.keyUp('up')

def press_down(approx_relase_time):
    pyautogui.keyDown('down')
    while True:
        if time.time() > approx_relase_time:
            break
    pyautogui.keyUp('down')
    
whole_screen = capture_screen(image_grab_bbox=None)
# Select region of interest then press ENTER
roi = cv2.selectROI("ROI", whole_screen) 

rex = Dino()
obstacle_list = []
secs_per_frame = 0

down_relase_timestamp = time.time()

while True:

    frame_start_timestamp = time.time()

    roi_screenshot = capture_screen(image_grab_bbox=(int(roi[0]), int(roi[1]), int(roi[0]+roi[2]), int(roi[1]+roi[3])))
    grayscale_game_img = cv2.cvtColor(roi_screenshot, cv2.COLOR_BGR2GRAY)
    _, game_thresh = cv2.threshold(grayscale_game_img, 127, 255,0)
    bgr_game_thresh = cv2.cvtColor(game_thresh, cv2.COLOR_GRAY2BGR)
    
    if game_thresh[0, 0] == 0:
        game_thresh = revert_colors(game_thresh)

    game_dino_contour = rex.approx_game_contour(game_thresh)

    if game_dino_contour is not None:

        rex.update_contour_properties()
        rex.update_obstacle_view_properties(game_thresh)
        rex.update_jump_properties()

        obstacle_list = update_obstacle_list(rex, obstacle_list, game_thresh)

        threatening_obstacle = rex.jump_or_duck_obstacle(obstacle_list, secs_per_frame)
        if threatening_obstacle != None:
            obstacle_width = threatening_obstacle.end_x - threatening_obstacle.start_x + 1
            if threatening_obstacle.bottom > rex.obstacle_view_max_y:
                thread_u = threading.Thread(target=press_up)
                thread_u.start()
            elif (time.time() > down_relase_timestamp and rex.top_point < threatening_obstacle.bottom 
                  and threatening_obstacle.pixels_per_sec > 0 and obstacle_width > rex.width / 2):
                distance = abs(threatening_obstacle.start_x - rex.left_most_point)
                obstacle_width = abs(threatening_obstacle.end_x - threatening_obstacle.start_x)
                approx_secs_over = (obstacle_width + distance) / threatening_obstacle.pixels_per_sec
                down_relase_timestamp = time.time() + approx_secs_over
                print(threatening_obstacle.end_x - threatening_obstacle.start_x, ',', threatening_obstacle.bottom - threatening_obstacle.top)
                thread_d = threading.Thread(target=press_down, args=(down_relase_timestamp,))
                thread_d.start()

        obstacle_count = len(obstacle_list)
        color_list = [(114, 97, 78), (36, 79, 36), (24, 51, 97), (41, 56, 73)]
        for i in range(obstacle_count):
            obstacle_i = obstacle_list[i]
            rec_top_left = (obstacle_i.start_x, obstacle_i.top)
            rec_bottom_right = (obstacle_i.end_x, obstacle_i.bottom)
            rec_color = color_list[i % len(color_list)]
            cv2.rectangle(bgr_game_thresh,rec_top_left, rec_bottom_right, rec_color, thickness=-1)

        cv2.imshow("Obstacle View", rex.obstacle_view_thresh)
    
    else:
        cv2.imshow("Obstacle View", game_thresh * 0)

    cv2.imshow("ROI", bgr_game_thresh)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    frame_end_timestamp = time.time()
    secs_per_frame = abs(frame_end_timestamp - frame_start_timestamp)