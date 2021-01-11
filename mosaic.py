# import the necessary packages
import numpy as np
import cv2


class Stitcher:
    def __init__(self, grid, main_channel):
        self.cols = grid[1]
        self.rows = grid[0]
        self.main_channel = main_channel

    def mosaic(self, well, ix):
        """
        Takes all images of one well. Reduces the size of images in main channel
        and calculates all homography matrices for every position.
        Then applies homographies to images in normal resolution in all channels.
        :param well: dictionary of images for one well in every channel
        :param i: index of well
        :return: tuple with stitched images for every channel
        """

        # start stitching
        print("[INFO] Begin stitching well {}".format(ix + 1))

        # create list of images in main channel with reduced size
        red = [cv2.resize(image, dsize=(400, 400)) for image in well[self.main_channel]]

        # calculate list of homographies from images in reduced size
        src = well[self.main_channel][0].shape
        dst = red[0].shape
        scale = (src[0]/dst[0], src[1]/dst[1])
        homographies = self.homography_list(red, scale)

        # if no keypoints were detected at any position homography_list
        # return None
        if homographies is None:
            print("[ERROR] Well {} can not be stitched!".format(ix))
            return None

        # iterate over all channels and apply stitching with predefined homography matrices
        stitched = {}

        for channel in well.keys():

            # initialize row and column
            row = 1
            col = 1

            # initialize homography index
            hom = 0

            for i, image in enumerate(well[channel]):

                if i == 0:
                    # create an empty background and insert the first
                    # image in the top left corner
                    print("[INFO] Stitching {} at position: row {} and column {}".format(channel, row, col))
                    b_shape = image.shape[0] * self.rows, image.shape[1] * self.cols
                    result = np.zeros(b_shape, dtype=np.uint8)
                    result[:image.shape[0], :image.shape[1]] = image

                else:
                    # give information about status
                    print("[INFO] Stitching {} at position: row {} and column {}".format(channel, row, col))

                    # stitch image to the background
                    result = self.stitch((result, image), M=homographies[hom])
                    hom = hom + 1

                # update row and col
                if col < 5:
                    col = col + 1
                else:
                    col = 1
                    row = row + 1

            # crop the stitched image
            stitched[channel] = self.crop_image(result)

        print("[INFO] Done stitching well {}".format(ix + 1))
        return stitched

    def homography_list(self, images, scale):
        """
        Takes small sized images in main channel of one well and
        returns a list of tuples containing matches, homography matrices
        and statuses for every position of the grid
        :param images: list of images
        :param i: index of well
        :param scale: scale factor from image with reduced size to original image
        :return list:
        """

        # store homographies in list
        homographies = []

        # initialize row and column
        row = 1
        col = 1

        for i, image in enumerate(images):

            if i == 0:
                # create an empty background and insert the first
                # image in the top left corner
                b_shape = image.shape[0] * self.rows, image.shape[1] * self.cols
                result = np.zeros(b_shape, dtype=np.uint8)
                result[:image.shape[0], :image.shape[1]] = image

            else:
                # find homography for current position
                M = self.get_homography([result, image])
                # give information about status
                if M is None:
                    print("[INFO] No keypoints detected at position: row {} and column {}".format(row, col))
                    return None
                else:
                    print("[INFO] Keypoints detected at position row: {} and column {}".format(row, col))

                # stitch image to the background
                result = self.stitch((result, image), M=M)

                # transform homography matrix and add it to list
                M = (M[0], self.transform_homography(M[1], image.shape, (int(image.shape[0] * scale[0]), int(image.shape[1] * scale[1]))), M[2])
                homographies.append(M)

            # update row and col
            if col < 5:
                col = col + 1
            else:
                col = 1
                row = row + 1

        return homographies

    def transform_homography(self, H, src_shape, dst_shape):
        '''
        Takes the homography matrix of a downsized image and transforms ist
        so that it can be applied to the image in original size
        :param H: 3 x 3 homography matrix
        :param src_shape: shape of downsized image
        :param dst_shape: shape of original image
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

    def stitch(self, images, M):
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

        # the size of the result equals size of imageA
        dsize = imageA.shape

        # create mask for imageB for warping
        imageB_mask = np.zeros(imageB.shape)
        imageB_mask[:] = 255

        # warp imageB and the mask using the homography matrix
        result = cv2.warpPerspective(imageB, H, dsize, flags=cv2.INTER_LINEAR)
        result_mask = cv2.warpPerspective(imageB_mask, H, dsize)
        # invert the warped mask
        result_mask = cv2.bitwise_not(np.uint8(result_mask))

        # delete the part of imageA where we want to add imageB
        masked_image = cv2.bitwise_and(np.uint8(imageA), result_mask)

        # add imageB to imageA at the correct position
        result = cv2.bitwise_or(np.uint8(result), masked_image)

        # cv2.imshow("test", result)
        # cv2.waitKey(0)

        # return the stitched image
        return result

    def get_homography(self, images, ratio=0.7, reprojThresh=7.0, color="grayscale"):
        """
        Takes two small sized images in main channel of one well and
        returns a tuple with matches, homography matrix and status
        :param images: list of two images
        :return tuple:
        """

        # unpack the images, then detect keypoints and extract
        # local invariant descriptors from them
        (imageA, imageB) = images
        (kpsA, featuresA) = self.detectAndDescribe(imageA, color=color)
        (kpsB, featuresB) = self.detectAndDescribe(imageB, color=color)

        # match features between the two images
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

        # if the match is None, then there aren't enough matched
        # keypoints to create a stitch
        if M is None:
            return None

        return M

    def detectAndDescribe(self, image, color):
        """
        Detects keypoints and corresponding descriptors from image
        :param image: numpy array
        :param color: string
        :return: keypoints and descriptors (features)
        """
        if color == "RGB":
            # convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif color == "BGR":
            # convert image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            # image is already grayscale
            gray = image

        # detect and extract features from the image
        descriptor = cv2.SIFT_create()
        (kps, features) = descriptor.detectAndCompute(gray, None)

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
        search_params = dict(checks=50)  # or pass empty dictionary

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
            (H, status) = cv2.findHomography(ptsB, ptsA, cv2.RANSAC,
                                             reprojThresh)
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
        return image[:row_end, 0:col_end]

    def equal_size(self, images):
        '''
        Takes a list of images and resizes them so that they all have the
        same dimensions
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
