# import the necessary packages
from mosaic import Stitcher
# from stitching.panorama import Stitcher
from imageloader import ImageLoader
import argparse
import cv2
import os

# construct the argument parse and parse the arguments
modes = "RGB", "BGR", "grayscale"
channels = "TexasRed", "DAPI", "FITC", "Cy3", "Cy5"
grid = (5, 5)
overlap = 20
main_channel = "TexasRed"
string = "{r} - {cc}"

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str, required=True,
                help="path to input directory of images to stitch")
ap.add_argument("-o", "--output", type=str, required=True,
                help="path to output directory for stitched image")
ap.add_argument('-gr', '--grid', type=int, default=grid, required=False, nargs="+",
                help="Image Grid that should be stitched together. Rows and Columns.")

# optional arguments
ap.add_argument("-s", "--string", type=str, required=False, default=string,
                help="String by which the well can be identified. Default: '{r} - {cc}'.")
ap.add_argument('-ch', '--channels', type=list, default=channels, required=False,
                help="Channels to load")
ap.add_argument("-m", "--mainchannel", type=str, required=False, default=main_channel,
                help="Main channel, that should be used for keypoint detection")
args = vars(ap.parse_args())

# Load images
# Dictionary of wells:
# First level: wells, second level: channels, third level: list of images as arrays
images = ImageLoader(path=args["images"], channels=args["channels"], string=args["string"])

# test weather the grid arguments satisfies requirements
grid = tuple(args["grid"])
if len(grid) > 2:
    raise Exception("[Argument Error] Two integers expected for grid argument: rows columns")


def main(images=images, grid=grid, main_channel=args["mainchannel"]):
    stitcher = Stitcher(grid=grid, main_channel=main_channel)

    for i, (well, channelDict) in enumerate(images.load_images()):
        result = stitcher.mosaic(channelDict, ix=i)

        for channel in result.keys():
            if result[channel] is not None:
                # cv2.imshow(str(i), results[well][main_channel])
                cv2.imwrite(os.path.join(args["output"], "{} {}.tif".format(well, channel)), result[channel])
                # cv2.waitKey(0)
            else:
                print("[INFO] No stitched image for well {} in {} channel".format(well, channel))


if __name__ == "__main__":
    main()
