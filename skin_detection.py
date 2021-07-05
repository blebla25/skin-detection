import numpy as np
import PIL.Image as Image
import cv2
import matplotlib.pyplot as plt

def skin_detection(slika):
    binary = np.zeros((slika.shape[0], slika.shape[1]))
    binary = binary.astype(np.bool)
    R = slika[:,:,0]
    G = slika[:,:,1]
    B = slika[:,:,2]
    r = np.divide(R, (R+B+G))
    g = np.divide(G, (R+B+G))
    b = np.divide(B, (R+B+G))
    tmp0 = np.divide(r, g)
    tmp1 = np.divide((r*b), ((r+b+g) ** 2))
    tmp2 = np.divide((r*g), ((r+b+g) ** 2))
    index0 = tmp0[:,:] > 1.185
    index1 = tmp1[:,:] > 0.107
    index2 = tmp2[:,:] > 0.112
    index = np.logical_and(index0, index1)
    final = np.logical_and(index, index2)
    binary[final] = True
    return binary.astype(np.uint8)


if __name__ == '__main__':
    image = plt.imread("guys.jpg")
    plt.imshow(image)
    plt.show()
    tmp = skin_detection(image)
    plt.imshow(tmp)
    plt.show()
