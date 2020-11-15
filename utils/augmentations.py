import numpy as np
import cv2

class PerspectiveTransform(object):
    """Approximate perspective transform by selecting 4 random points at the corners of the image
       and using warp transformation."""

    def __init__(self, pt=0.1, keep_size=True):
        assert isinstance(pt, float)
        assert isinstance(keep_size, bool)
        self.pt = pt
        self.keep_size = keep_size

    def __call__(self, sample):
        image = sample
        height = image.shape[0]
        width = image.shape[1]

        # 0 - 3 -> Different tilts
        choices = [0,1,2,3]
        choice = np.random.choice(choices)

        if choice == 0:
          # Where both the top corners are closer
          pix_x_0 = int(((1 + np.random.random()) / 2 * self.pt) * width) 
          pix_x_1 = int((1 -((1 + np.random.random()) / 2 * self.pt)) * width)
          pix_x_2 = int((np.random.random() * self.pt / 2) * width)
          pix_x_3 = int((1 - (np.random.random() * self.pt / 2)) * width)
          pix_y_0 = int((np.random.random() * self.pt / 2) * height)
          pix_y_1 = int((np.random.random() * self.pt / 2) * height)
          pix_y_2 = int((1 - (np.random.random() * self.pt / 2)) * height)
          pix_y_3 = int((1 - (np.random.random() * self.pt / 2)) * height)       
        elif choice == 1:
          # Where both the bottom corners are closer
          pix_x_0 = int((np.random.random() * self.pt / 2) * width)
          pix_x_1 = int((1 - (np.random.random() * self.pt / 2)) * width)
          pix_x_2 = int(((1 + np.random.random()) / 2 * self.pt) * width) 
          pix_x_3 = int((1 -((1 + np.random.random()) / 2 * self.pt)) * width)
          pix_y_0 = int((np.random.random() * self.pt / 2) * height)
          pix_y_1 = int((np.random.random() * self.pt / 2) * height)
          pix_y_2 = int((1 - (np.random.random() * self.pt / 2)) * height)
          pix_y_3 = int((1 - (np.random.random() * self.pt / 2)) * height)
        elif choice == 2:
          # Where both the left corners are closer
          pix_x_0 = int((np.random.random() * self.pt / 2) * width) 
          pix_x_1 = int((1 - (np.random.random() * self.pt / 2)) * width)
          pix_x_2 = int((np.random.random() * self.pt / 2) * width)
          pix_x_3 = int((1 - (np.random.random() * self.pt / 2)) * width)
          pix_y_0 = int(((1 + np.random.random()) / 2 * self.pt) * height)
          pix_y_1 = int((np.random.random() * self.pt / 2) * height)
          pix_y_2 = int((1 -((1 + np.random.random()) / 2 * self.pt)) * height)
          pix_y_3 = int((1 - (np.random.random() * self.pt / 2)) * height)
        elif choice == 3:
          # Where both the right corners are closer
          pix_x_0 = int((np.random.random() * self.pt / 2) * width) 
          pix_x_1 = int((1 - (np.random.random() * self.pt / 2)) * width)
          pix_x_2 = int((np.random.random() * self.pt / 2) * width)
          pix_x_3 = int((1 - (np.random.random() * self.pt / 2)) * width)
          pix_y_0 = int((np.random.random() * self.pt / 2) * height)
          pix_y_1 = int(((1 + np.random.random()) / 2 * self.pt) * height)
          pix_y_2 = int((1 - (np.random.random() * self.pt / 2)) * height)
          pix_y_3 = int((1 -((1 + np.random.random()) / 2 * self.pt)) * height)

        pix_warp = np.float32([
                                [pix_x_0, pix_y_0],
                                [pix_x_1, pix_y_1],
                                [pix_x_2, pix_y_2],
                                [pix_x_3, pix_y_3],
                              ])
        
        pix_tfm = np.float32([
                                [0, 0],
                                [width, 0],
                                [0, height],
                                [width, height],
                             ])

        matrix = cv2.getPerspectiveTransform(pix_warp, pix_tfm)
        result = cv2.warpPerspective(image, matrix, (width, height))
        return result