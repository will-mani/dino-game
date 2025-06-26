import cv2
import numpy as np
import time

class Dino:
    def __init__(self):
        self.right_most_point = None
        self.left_most_point = None
        self.width = None

        self.prev_top_point = None
        self.top_point = None
        self.bottom_point = None
        self.height = None

        self.model_contour = self.generate_model_contour()
        self.contour = None

        self.last_off_ground_timestamp = 0
    
        self.obstacle_view_min_x= None

        self.obstacle_view_min_height_percentage = 0.4
        self.obstacle_view_max_y = None
        self.obstacle_view_max_height_percentage = 1.2
        self.obstacle_view_min_y = None

        self.obstacle_view_thresh = None

        self.obstacle_velocity_multiplier = 1
        self.distance_width_multiplier = 2


    def generate_model_contour(self): 
        model_dino_img = cv2.imread('imgs/dino.png', cv2.IMREAD_GRAYSCALE)
        model_dino_img = cv2.resize(model_dino_img, (0, 0), fx = 0.5, fy = 0.5)
        _, thresh = cv2.threshold(model_dino_img, 127, 255,0)
        contours, _ = cv2.findContours(thresh, 2, 1)
        model_dino_contour = contours[2]
        return model_dino_contour


    def approx_game_contour(self, game_thresh):
        game_contours, _ = cv2.findContours(game_thresh, 2, 1)
        min_difference = float('inf')
        for potential_contour in game_contours:
            difference = cv2.matchShapes(self.model_contour, potential_contour, 1, 0.0)
            if difference < min_difference:
                self.contour = potential_contour
                min_difference = difference
        
        if min_difference > 0.1:
            self.contour = None

        return self.contour
    
    def update_contour_properties(self):
        reshaped_contour = np.reshape(self.contour, (self.contour.shape[0], self.contour.shape[2]))
        
        self.right_most_point = max(reshaped_contour[:, 0])
        self.left_most_point = min(reshaped_contour[:, 0])
        self.width = self.right_most_point - self.left_most_point

        self.prev_top_point = self.top_point
        self.top_point = min(reshaped_contour[:, 1])
        self.bottom_point = max(reshaped_contour[:, 1])
        self.height = self.bottom_point - self.top_point


    def is_defenitely_on_ground(self):
        if self.prev_top_point == None:
            self.prev_top_point = self.top_point
        delta_y = abs(self.prev_top_point - self.top_point)
        if delta_y > 0:
            self.last_off_ground_timestamp = time.time()
            return False
        secs_on_ground = time.time() - self.last_off_ground_timestamp
        return secs_on_ground > 2
    
    
    def update_obstacle_view_properties(self, game_thresh):
        if self.is_defenitely_on_ground():
            self.obstacle_view_min_x = self.right_most_point
            self.obstacle_view_max_y = self.bottom_point - int(self.height * self.obstacle_view_min_height_percentage)
            self.obstacle_view_min_y = self.bottom_point - int(self.height * self.obstacle_view_max_height_percentage)

        self.obstacle_view_thresh = game_thresh.copy()
        self.obstacle_view_thresh[:, :self.obstacle_view_min_x] = 127
        self.obstacle_view_thresh[self.obstacle_view_max_y + 1:, :] = 127
        self.obstacle_view_thresh[:self.obstacle_view_min_y, :] = 127
        cv2.drawContours(self.obstacle_view_thresh, [self.contour], 0, 255, -1)


    def jump_or_duck_obstacle(self, obstacle_list, secs_per_frame):
        obstacle_count = len(obstacle_list)
        for i in range(min(2, obstacle_count)):
            obstacle = obstacle_list[i]
            adjusted_obstacle_velocity = obstacle.pixels_per_sec * self.obstacle_velocity_multiplier
            if (obstacle.start_x - (adjusted_obstacle_velocity * secs_per_frame)) <= (self.right_most_point + (self.width * self.distance_width_multiplier)):
                return obstacle
              
        return None