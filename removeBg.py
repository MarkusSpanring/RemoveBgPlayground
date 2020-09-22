import cv2 as cv
import numpy as np

# Very simple attempt to remove background from an image
# Works ok but is a bit fine-tuned....


def coloredSplit(img, channel="ch1"):
    """Color channels of image. Just needed for debugging"""
    zero = np.zeros(img.shape, dtype=img.dtype)
    if channel == "ch1":
        return cv.merge((img, zero, zero))
    if channel == "ch2":
        return cv.merge((zero, img, zero))
    if channel == "ch3":
        return cv.merge((zero, zero, img))


def blurAndThreshold(channel):
    """try to make boundary between object and bkg with similar color"""
    channel = cv.GaussianBlur(channel, (15, 15), cv.BORDER_DEFAULT)
    ret, channel = cv.threshold(channel, 110, 255, cv.THRESH_BINARY)
    return channel


def splitAndPreprocess(img, debug=False):
    """split img into individual channels and apply some preprocessing"""
    ch1, ch2, ch3 = cv.split(img)

    ch1 = blurAndThreshold(ch1)
    ch2 = blurAndThreshold(ch2)
    ch3 = blurAndThreshold(ch2)

    if debug:
        thresstack = np.hstack([coloredSplit(ch1, "ch1"),
                                coloredSplit(ch2, "ch2"),
                                coloredSplit(ch3, "ch3"),
                                cv.merge((ch1, ch2, ch3))])
        cv.imwrite("split.png", thresstack)

    return ch1, ch2, ch3


def getMaskFromLargestContour(channel):
    """Get contour from image and return largest one as mask."""

    contours, hierarchy = cv.findContours(channel, cv.RETR_LIST,
                                          cv.CHAIN_APPROX_NONE)
    max_ar = 0
    max_cnt = 0
    for i, cnt in enumerate(contours):
        if cv.contourArea(cnt) > max_ar:
            max_ar = cv.contourArea(cnt)
            max_cnt = cnt

    x, y = channel.shape
    mask = np.zeros((x, y, 4), dtype=channel.dtype)
    return cv.drawContours(mask, [max_cnt], 0, (255, 255, 255, 255), -1)


def addAlpha(img):
    """Add alpha to image"""
    ch1, ch2, ch3 = cv.split(img)
    alpha_channel = np.full(ch1.shape, 255, dtype=ch1.dtype)
    return cv.merge((ch1, ch2, ch3, alpha_channel))


def removeBg(img, mask, debug=False):
    """Remove background by masking out largest contour"""
    result = cv.bitwise_and(img, mask)

    if debug:
        cv.imwrite("original.png", img)
        cv.imwrite("removed.png", result)

    double = np.vstack([img, result])
    cv.imwrite("sidebyside.png", double)


def main():
    debug = False
    img = cv.imread('new_years_eve_cat.jpg')
    ch1, ch2, ch3 = splitAndPreprocess(img, debug)
    mask = getMaskFromLargestContour(ch1)
    img_BGRA = addAlpha(img)
    removeBg(img_BGRA, mask, debug)


if __name__ == '__main__':
    main()
