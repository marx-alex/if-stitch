import os
import cv2
from imutils import paths
from itertools import groupby
import re

class ImageLoader:
    def __init__(self, path, channels, string):
        self.path = path
        self.channels = channels
        self.string = string

    def load_images(self):
        # Loading images
        print("[INFO] loading images...")
        inp = os.path.normpath(self.path)
        imagePaths = sorted(list(paths.list_images(inp)))

        # use string argument and change it to regular expression
        self.string = self.string.replace("{r}", "[A-H]")
        self.string = self.string.replace("{cc}", "[0-1][0-9]")
        # string will be like: "[A-H] - [0-1][0-9]"

        # use the string to group the images by wells
        keyf = lambda text: re.findall(self.string, text)[0]
        wells = [(gr, list(items)) for gr, items in groupby(imagePaths, key=keyf)]

        # create a dictionary of images by wells and channels
        # imageDict = {}
        # iterate through grouped wells
        for well, images in wells:
            channelDict = {}
            # iterate through valid channels
            for channel in self.channels:

                # add channel to dictionary if found in file
                ch = [cv2.imread(imagePath, 0) for imagePath in images if str(channel) in imagePath]

                if len(ch) >= 1:
                    channelDict[channel] = ch

            # imageDict[well] = channelDict
            yield well, channelDict

        # return imageDict

