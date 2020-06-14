import numpy as np
from cv2 import cv2


def empty(self):
    pass

class RUNME:

    def run(self):
        img = cv2.imread('test2.png',1)
        width = img.shape[1]
        height = img.shape[0]
        image = img
        img = cv2.resize(img, (900,200))

        self.windowName = "window"
        cv2.namedWindow(self.windowName)

        # creating Trackbars, these are linked by their name.
        cv2.createTrackbar("H_min", self.windowName, 0, 180, empty)
        cv2.createTrackbar("H_max", self.windowName, 180, 180, empty)

        cv2.createTrackbar("S_min", self.windowName, 0, 255, empty)
        cv2.createTrackbar("S_max", self.windowName, 255, 255, empty)

        cv2.createTrackbar("V_min", self.windowName, 50, 255, empty)
        cv2.createTrackbar("V_max", self.windowName, 255, 255, empty)

        cv2.createTrackbar("alpha", self.windowName, 0, 500, empty)
        cv2.createTrackbar("beta", self.windowName, 0, 50, empty)

        while True:

            Hmin = cv2.getTrackbarPos("H_min", self.windowName)
            Hmax = cv2.getTrackbarPos("H_max", self.windowName)
            Smin = cv2.getTrackbarPos("S_min", self.windowName)
            Smax = cv2.getTrackbarPos("S_max", self.windowName)
            Vmin = cv2.getTrackbarPos("V_min", self.windowName)
            Vmax = cv2.getTrackbarPos("V_max", self.windowName)
            alpha = cv2.getTrackbarPos("alpha", self.windowName)
            beta = cv2.getTrackbarPos("beta", self.windowName)


            hsv = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)

            lower_blue = np.array([Hmin, Smin, Vmin])
            upper_blue = np.array([Hmax, Smax, Vmax])
            mask = cv2.inRange(hsv, lower_blue, upper_blue)

            cv2.imshow(self.windowName, mask)
            cv2.imshow("orig",img)
            mask = cv2.resize(mask,(width,height))

            gammma = self.gammaCorrection(image,alpha,beta)
            cv2.imshow("image", image)
            #cv2.imshow("gammma", gammma)
            cv2.imshow("mask",mask)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

    def gammaCorrection(self, image, alpha, beta):

        alpha = float(alpha/100)
        beta = int(beta)

        new_image = image.copy()
        print(image.shape[2])

        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                for c in range(image.shape[2]):
                    new_image[y, x, c] = np.clip(alpha * image[y, x, c] + beta, 0, 255)

        return new_image