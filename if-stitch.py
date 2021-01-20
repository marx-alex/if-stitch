# import the necessary packages
import numpy as np
import cv2
import os
import argparse
from imutils import paths
from itertools import groupby
import re


class Stitcher:
    def __init__(self, grid, main_channel):
        self.cols = grid[1]
        self.rows = grid[0]
        self.main_channel = main_channel
        self.redsize = (400, 400)

    def __call__(self, well, ix, ratio, reprojThresh):
        """
        Takes all images of one well. Reduces the size and depth of images in main channel
        and calculates all homography matrices for every position.
        Then applies homographies to images in normal resolution and depth in all channels.
        :param well: dictionary of images for one well in every channel
        :param i: index of well
        :return: tuple with stitched images for every channel
        """

        # start stitching
        print("[INFO] Begin stitching well {}".format(ix + 1))

        # test if images contain unsigned integers
        if not np.issubdtype(well[self.main_channel][0].dtype, np.integer):
            raise Exception("[ERROR] Images do not contain integers")
        if not np.issubdtype(well[self.main_channel][0].dtype, np.unsignedinteger):
            raise Exception("[ERROR] Images do not contain unsigned integers")

        # get maximal value of used image depth
        max_value = float(np.iinfo(well[self.main_channel][0].dtype).max)
        # create list of images in main channel with reduced size and in uint8
        red = [cv2.convertScaleAbs(cv2.resize(image, dsize=self.redsize), alpha=(255.0 / max_value)) for image in
               well[self.main_channel]]

        # calculate list of homographies from images in reduced size
        src = well[self.main_channel][0].shape
        dst = red[0].shape
        scale = (src[0] / dst[0], src[1] / dst[1])
        homographies = self.mosaic(red, scale=scale, ratio=ratio, reprojThresh=reprojThresh)

        # iterate over all channels and apply stitching with predefined homography matrices
        stitched = {}

        for channel in well.keys():
            # if no keypoints were detected at any position in homographies is None
            if homographies is None:
                print("[ERROR] Well {} in channel {} can not be stitched!".format(ix, channel))
                stitched[channel] = None
            # else apply homography to images
            else:
                stitched[channel] = self.mosaic(well[channel], channel=channel, hom=homographies, ratio=ratio, reprojThresh=reprojThresh)

        # start stitching
        print("[INFO] Done stitching well {}".format(ix + 1))
        return stitched

    def mosaic(self, images, ratio, reprojThresh, scale=None, channel=None, hom=False):
        """
        Stitches images of a well, first as row panoramas then the rows to a whole image.
        If hom is given, applies homographies to images and returns the stitched image.
        If no hom is given, calculates homographies at every position and return them as list.
        :type ratio: integer
        :param reprojThresh: integer
        :param images: list of numpy arrays
        :param scale: tuple of two float numbers
        :param channel: string
        :param hom: list of 3x3 numpy arrays
        :return: stitched image (array) or homography list (list of 3x3 numpy arrays)
        """
        if not hom:
            # store homographies in list
            homographies = []
        else:
            hom_ix = 0

        # initialize row and column
        row = 1
        col = 1

        # store stitched rows in list
        stitched_rows = []

        for i, image in enumerate(images):

            if col == 1:
                # start with the first image
                if hom:
                    print("[INFO] Stitching {} at position: row {} and column {}".format(channel, row, col))
                result = image

            else:
                if hom:
                    # give information about status
                    print("[INFO] Stitching {} at position: row {} and column {}".format(channel, row, col))

                    # stitch image to the preceding one
                    result = self.stitch((result, image), M=hom[hom_ix], pos=(row, col))
                    hom_ix = hom_ix + 1

                else:
                    # find homography for current position
                    M = self.get_homography([result, image], pos=(row, col), ratio=ratio, reprojThresh=reprojThresh)
                    # return None if there are not enough keypoints and give info
                    if M is None:
                        print("[INFO] No keypoints detected at position: row {} and column {}".format(row, col))
                        return None
                    else:
                        # give info about current status
                        print("[INFO] Keypoints detected at position: row {} and column {}".format(row, col))

                    # stitch image to the background
                    result = self.stitch((result, image), M=M, pos=(row, col))

                    # transform homography matrix and add it to list
                    M = (M[0], self.transform_homography(M[1], image.shape, (
                        int(image.shape[0] * scale[0]), int(image.shape[1] * scale[1]))), M[2])
                    homographies.append(M)

            if col == self.cols:
                # append stitched_rows by panorama of one row
                stitched_rows.append(result)

            # update row and col
            if col < self.cols:
                col = col + 1
            else:
                col = 1
                row = row + 1

        # initialize rows
        row = 1

        # iterate over all panorama images of the rows
        for panorama in stitched_rows:

            if row == 1:
                if hom:
                    print("[INFO] Stitching {} at position: row {}".format(channel, row))
                # put the panorama on a black background and move it a bit from the corner
                result = np.zeros((int(panorama.shape[0] * 1.5), int(panorama.shape[1] * 1.5)), dtype=panorama.dtype)
                start = (int(panorama.shape[0] / 4), int(panorama.shape[1] / 4))
                result[start[0]:start[0] + panorama.shape[0], start[1]:start[1] + panorama.shape[1]] = panorama

            else:
                if hom:
                    print("[INFO] Stitching {} at position: row {}".format(channel, row))
                    # stitch image to the background
                    result = self.stitch((result, panorama), M=hom[hom_ix], pos=(row, 0))
                    hom_ix = hom_ix + 1

                else:
                    # stitch image to the background
                    M = self.get_homography([result, panorama], pos=(row, 0), ratio=ratio, reprojThresh=reprojThresh)
                    # return None if there are not enough keypoints in panorama and give info
                    if M is None:
                        print("[INFO] No keypoints detected at position: row {}".format(row))
                        return None
                    # give info about status
                    else:
                        print("[INFO] Keypoints detected at position: row {}".format(row))

                    # stitch row to the row above
                    result = self.stitch((result, panorama), M=M, pos=(row, 0))

                    # transform homography matrix and add it to list
                    M = (M[0], self.transform_homography(M[1], panorama.shape, (
                        int(panorama.shape[0] * scale[0]), int(panorama.shape[1] * scale[1]))), M[2])
                    homographies.append(M)

            # update row
            row = row + 1

        if hom:
            return self.crop_image(result)
        else:
            return homographies

    def transform_homography(self, H, src_shape, dst_shape):
        '''
        Takes the homography matrix of a downsized image and transforms it
        so that it can be applied to the image in original size
        :param H: 3 x 3 homography matrix
        :param src_shape: tuple, shape of downsized image
        :param dst_shape: tuple, shape of original image
        :return: 3 x 3 matrix
        '''

        # get matrix of equivalent points in both images
        src = np.array([[0, 0],
                        [src_shape[1], 0],
                        [0, src_shape[0]],
                        [src_shape[1], src_shape[0]]], dtype=np.float32)

        dst = np.array([[0, 0],
                        [dst_shape[1], 0],
                        [0, dst_shape[0]],
                        [dst_shape[1], dst_shape[0]]], dtype=np.float32)

        # get a transformation matrix
        trans = cv2.getPerspectiveTransform(src, dst)
        trans_inv = cv2.getPerspectiveTransform(dst, src)
        trans = trans.astype('float64')
        trans_inv = trans_inv.astype('float64')

        # apply transformation to all homographies
        H_trans = np.matmul((np.matmul(trans, H)), trans_inv)

        return H_trans

    def stitch(self, images, M, pos):
        """
        Takes to images, stitches them together and return the stitched image
        :param images: list of two images
        :param M: tuple of match, homography matrix and status
        :param stitch_rows: True for vertical stitching
        :return image:
        """

        # unpack images
        imageA, imageB = images
        # unpack homography information
        (matches, H, status) = M

        # size of the results depends on weather rows or columns are stitched
        row, col = pos
        if col == 0:
            # rows are stitched together
            dsize = (imageA.shape[1], imageA.shape[0] + imageB.shape[0])
        else:
            dsize = (imageA.shape[1] + imageB.shape[1], imageA.shape[0])

        # create mask for imageB for warping
        imageB_mask = np.zeros(imageB.shape, dtype=imageB.dtype)
        imageB_mask[:] = np.iinfo(imageB.dtype).max

        # warp imageB and the mask using the homography matrix
        result = cv2.warpPerspective(imageB, H, dsize, flags=cv2.INTER_LINEAR)
        result_mask = cv2.warpPerspective(imageB_mask, H, dsize)
        # invert the warped mask
        _, result_mask = cv2.threshold(result_mask, np.iinfo(result_mask.dtype).max - 1,
                                       np.iinfo(result_mask.dtype).max, cv2.THRESH_BINARY)  # new
        result = cv2.bitwise_and(result, result_mask)   # new
        result_mask = cv2.bitwise_not(result_mask)

        # delete the part of imageA where we want to add imageB
        background = np.zeros((dsize[1], dsize[0]), dtype=imageB.dtype)
        background[:imageA.shape[0], :imageA.shape[1]] = imageA
        masked_image = cv2.bitwise_and(background, result_mask)

        # add imageB to imageA at the correct position
        result = cv2.bitwise_or(result, masked_image)

        return result

    def get_homography(self, images, ratio, reprojThresh, pos=None):
        """
        Takes two small sized images in main channel of one well and
        returns a tuple with matches, homography matrix and status
        :param images: list of two images
        :return tuple:
        """

        # unpack the images, then detect keypoints and extract
        # local invariant descriptors from them
        (imageA, imageB) = images

        # detect keypoints
        row, col = pos
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # match features between the two images
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio=ratio, reprojThresh=reprojThresh)

        # if the match is None, then there aren't enough matched
        # keypoints to create a stitch
        if M is None:
            return None

        # change homography matrix so that the warped
        # imageB will have the same dimensions as before
        H = self.map_homography(M[1], imageB.shape, pos)

        return M[0], H, M[2]

    def map_homography(self, H, dsize, pos):
        """
        Takes a 3x3 homography matrix and changes it so that
        the warped image will be a paralellogram
        :param H: 3x3 numpy array, homography matrix
        :param dsize: tuple, size of the image to warp
        :param pos: tuple, row and column
        :return: 3x3 numpy array, homography matrix
        """
        # get fix corners of image warped with H
        # top left
        point_tl = np.array([[0], [0], [1]])
        tl = H.dot(point_tl).ravel()[:2]
        # bottom left
        point_bl = np.array([[0], [dsize[0]], [1]])
        bl = H.dot(point_bl).ravel()[:2]

        # get source corners of image
        src = np.array([[0, 0],
                        [dsize[1], 0],
                        [0, dsize[0]],
                        [dsize[1], dsize[0]]], dtype=np.float32)

        # extract position
        row, col = pos
        # get desired destination corners depending on position
        if col != 0:
            # if images are stitched in rows as panoramas only the left points are fixed
            dist = np.linalg.norm(tl - bl)

            dst = np.array([[tl[0], tl[1]],
                            [tl[0] + dist, tl[1]],
                            [bl[0], bl[1]],
                            [bl[0] + dist, bl[1]]], dtype=np.float32)

            # calculate new homography
            H = cv2.getPerspectiveTransform(src, dst)

            return H
        # do nothing if rows are stitched
        return H

    def detectAndDescribe(self, image, mask=None):
        """
        Detects keypoints and corresponding descriptors from image
        :param image: numpy array
        :return: keypoints and descriptors (features)
        """

        # detect and extract features from the image
        descriptor = cv2.SIFT_create()
        if mask is not None:
            (kps, features) = descriptor.detectAndCompute(image, mask=mask)
        else:
            (kps, features) = descriptor.detectAndCompute(image, None)

        # return a tuple of keypoints and features
        return kps, features

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        """
        Takes keypoints and descriptors from two images and matches them. Then calculates
        a homography matrix from matched keypoints.
        :param kpsA: keypoints ImageA
        :param kpsB: keypoints imageB
        :param featuresA: descriptors ImageA
        :param featuresB: descriptors ImageB
        :param ratio: ratio to extract good matches
        :param reprojThresh: threshold to find outliers in keypoints
        :return: homography matrix
        """
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=100)  # or pass empty dictionary

        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []
        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append(m[0])

        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points

            # Extract location of good matches
            ptsA = np.zeros((len(matches), 2), dtype=np.float32)
            ptsB = np.zeros((len(matches), 2), dtype=np.float32)

            for i, m in enumerate(matches):
                ptsA[i, :] = kpsA[m.queryIdx].pt
                ptsB[i, :] = kpsB[m.trainIdx].pt

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, reprojThresh)

            # return the matches along with the homography matrix
            # and status of each matched point
            return matches, H, status
        # otherwise, no homograpy could be computed
        return print("[INFO] Not enough keypoints")

    def crop_image(self, image, tol=0):
        """
        Crops horizontal and vertical black lines from image
        """
        # image is 2D image data
        # tol  is tolerance
        mask = image > tol
        m, n = image.shape
        mask0, mask1 = mask.any(0), mask.any(1)
        col_start, col_end = mask0.argmax(), n - mask0[::-1].argmax() - 1
        row_start, row_end = mask1.argmax(), m - mask1[::-1].argmax() - 1

        return image[row_start:row_end, col_start:col_end]

    def equal_size(self, images):
        '''
        Takes a list of images and resizes them so that they all have the
        same dimensions
        This function is currently not used
        :param images: list of images
        :return: list of images with same dimensions
        '''

        # get dimensions of all images
        dims = np.array([image.shape for image in images])
        # get the minimum values for dimensions
        mins = tuple(np.amin(dims, axis=0))

        output = []
        # apply new dimensions for all images
        for image in images:
            image = cv2.resize(image, (mins[1], mins[0]), interpolation=cv2.INTER_AREA)
            output.append(image)

        return output


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

        # iterate through grouped wells
        for well, images in wells:
            channelDict = {}
            # iterate through valid channels
            for channel in self.channels:

                # add channel to dictionary if found in file
                ch = [cv2.imread(imagePath, cv2.IMREAD_ANYDEPTH) for imagePath in images if
                      str(channel) in imagePath]

                if len(ch) >= 1:
                    channelDict[channel] = ch

            yield well, channelDict


def main():
    # construct the argument parse and parse the arguments
    channels = "TexasRed", "DAPI", "FITC", "Cy3", "Cy5"
    main_channel = "TexasRed"
    string = "{r} - {cc}"
    ratio= 0.6
    reprojThresh = 7.0

    ap = argparse.ArgumentParser()
    # required arguments
    ap.add_argument("-i", "--images", type=str, required=True,
                    help="path to input directory of images to stitch")
    ap.add_argument("-o", "--output", type=str, required=True,
                    help="path to output directory for stitched image")
    ap.add_argument('-gr', '--grid', type=int, required=True, nargs="+",
                    help="Image Grid that should be stitched together. Rows and Columns.")

    # optional arguments
    ap.add_argument("-s", "--string", type=str, required=False, default=string,
                    help="String by which the well can be identified. Default: '{r} - {cc}'.")
    ap.add_argument('-ch', '--channels', type=list, default=channels, required=False,
                    help="Channels to load")
    ap.add_argument("-m", "--mainchannel", type=str, required=False, default=main_channel,
                    help="Main channel, that should be used for keypoint detection")
    ap.add_argument('-r', '--ratio', type=int, required=False, default=ratio,
                    help="Lowe's ratio to filter matches")
    ap.add_argument('-t', '--thresh', type=int, required=False, default=reprojThresh,
                    help="Reprojection threshold RANSAC algorithm")
    # parse arguments
    args = vars(ap.parse_args())

    # initialize imageLoader
    images = ImageLoader(path=args["images"], channels=args["channels"], string=args["string"])

    # test weather the grid arguments satisfies requirements
    grid = tuple(args["grid"])
    if len(grid) > 2:
        raise Exception("[Argument Error] Two integers expected for grid argument: rows columns")

    stitcher = Stitcher(grid=grid, main_channel=args["mainchannel"])

    # iterate over the wells in a directory
    # only one well at a time is loaded into memory
    for i, (well, channelDict) in enumerate(images.load_images()):
        result = stitcher(channelDict, ix=i, ratio=args["ratio"], reprojThresh=args["thresh"])

        for channel in result.keys():
            if result[channel] is not None:
                # cv2.imshow(str(i), result[channel])
                print("[INFO] Saving well {} in channel {}".format(well, channel))
                cv2.imwrite(os.path.join(args["output"], "{} {}.tif".format(well, channel)), result[channel])
                # cv2.waitKey(0)
            else:
                print("[INFO] No stitched image for well {} in {} channel".format(well, channel))


if __name__ == "__main__":
    main()
