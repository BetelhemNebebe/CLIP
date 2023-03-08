import numpy as np
import torch
import os
import cv2
from PIL import Image

import clip
import wav2clip
from moviepy.editor import VideoFileClip

class Load_data():
    def __init__(self, dir_path):
        super().__init__()
        self.dir_path = dir_path


    def get_file_list(self):
        """Takes directory path and returns list of paths to each video"""
        file_list = []
        print("Dir path: ", self.dir_path)

        # Iterate directory
        for path in os.listdir(self.dir_path):
            # check if current path is a file
            if os.path.isfile(os.path.join(self.dir_path, path)):
                file_list.append(os.path.join(self.dir_path, path))
        print("Video files are read!")
                
        return file_list


    def extract_avg_frame(self, video_path):
        """Takes a single video and returns the mean of the frames"""
        video = cv2.VideoCapture(video_path)

        success = True
        count = 0
        sum_frame = 0

        while success:
            success, frames = video.read()
            if success:
                sum_frame = sum_frame + frames
                #print("Sum frames: ", sum_frame)
                count = count + 1
            else:
                break

        mean_frame = sum_frame/count
        print("Video frames are sucessfully extracted!")
        return mean_frame

    def extract_frames(self, video_path):
        """Takes a single video and returns the frames"""
        video = cv2.VideoCapture(video_path) #video.get(cv2.CAP_PROP_FPS) => 25.0

        success = True
        frames_list = []
        count = 0

        while success:
            success, frames = video.read()
            if success and count < 125: #taking only 125 frames: 5 second clip * 25 fps
                frames_list.append(frames)
                count = count + 1

            else:
                break

        print("Video frames are sucessfully extracted!")
        return frames_list


    def extract_audio(self, video_path):
        """Takes the path to a single video and returns the extracted audio"""
        video = VideoFileClip(video_path)
        video_clip = video.subclip(0, 5) #take only the first 5 seconds
        audio = video_clip.audio #video_clip.audio.fps => 44100 Hz
        print("Audio is sucessfully extracted!")
        return audio