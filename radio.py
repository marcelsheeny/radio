import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

# fixed params
im_path = 'bike.png'
im_orig = cv2.imread(im_path, -1)
class_name = 'bike'
azi_res = 1.5
curr_range = 3.8
bg_mean = 13.6075
bg_stddev = 23.73201305726086
thresh = 195
title = 'Radar Data Augmentation'


# variable params initialization
# simulate new range
new_range = 3.8

# simulate speckle noise
stddev = 0

# simulate different background level
init_bg_mean = 0

# simulate different rotations
rot = 0
tx = 0
ty = 0


def nothing(x):
    pass


def attenuation_trolley(x):
    return -3.96 * x


def attenuation_sign(x):
    return -7.59 * x


def attenuation_mannequin(x):
    return -3.89 * x


def attenuation_dog(x):
    return -3.70 * x


def attenuation_bike(x):
    return -3.99 * x


def attenuation_cone(x):
    return -5.08 * x


def height_function(x):
    return 1.3333 * x


def width_function(x):
    return 2.460 * x


def change_res(im_orig, curr_range, new_range, azi_res=1.5):
    diff_range = new_range - curr_range
    # change resolution
    azimuth_res = 2.0 * diff_range * math.sin(math.radians(azi_res / 2.0))

    range_res = 0.0075
    scale_width = 10.0 * (azimuth_res / range_res)
    small_im = cv2.resize(
        im_orig, (int(im_orig.shape[1] - scale_width), im_orig.shape[0]), cv2.INTER_NEAREST)
    new_img = cv2.resize(
        small_im, (im_orig.shape[1], im_orig.shape[0]), cv2.INTER_NEAREST)
    return new_img


def attenuation_dataaug(im_orig, class_name, curr_range, new_range):
    im_orig = im_orig.astype(np.float)
    # cfar_im = (im_orig > 175) * 255.0
    cfar_im = (im_orig > thresh) * 255.0
    # cfar_im = cfar2d(im_orig, 200, 10, 0.25)
    diff_range = new_range - curr_range
    if (class_name == 'trolley' or class_name == 'object'):
        ratio = attenuation_mannequin(diff_range)
    elif(class_name == 'bike'):
        ratio = attenuation_bike(diff_range)
    elif(class_name == 'cone'):
        ratio = attenuation_cone(diff_range)
    elif(class_name == 'dog'):
        ratio = attenuation_dog(diff_range)
    elif(class_name == 'mannequin'):
        ratio = attenuation_mannequin(diff_range)
    elif(class_name == 'sign'):
        ratio = attenuation_sign(diff_range)
    elif(class_name == 'background'):
        ratio = 0

    foreground = np.multiply(cfar_im == 255, im_orig + ratio)
    background = np.multiply(cfar_im == 0, im_orig)

    foreground = np.clip(foreground, 0, 255)

    new_img = foreground + background

    return new_img


def speckle_noise_dataaug(im_orig, stddev):
    im_orig = im_orig
    gauss = stddev * \
        np.random.randn(im_orig.shape[0], im_orig.shape[1]) + 1.0
    gauss = gauss.reshape(im_orig.shape[0], im_orig.shape[1])
    noisy = (np.clip(im_orig.astype(np.float)
                     * gauss, 0, 255))
    return noisy


def fill_zeros_with_gaussian_noise(im, bg_mean, bg_stddev):
    background_mask = (im < 20).astype(np.float)
    gauss = bg_stddev * \
        np.random.randn(im.shape[0], im.shape[1]) + bg_mean
    gauss = gauss.reshape(im.shape[0], im.shape[1]) + 128
    background = np.multiply(background_mask, gauss)
    new_img = im.astype(
        np.float) + background.astype(np.float)
    return new_img


def new_bg(im_orig, mean):
    foreground = ((im_orig > thresh) * im_orig).astype(np.float)
    background = ((im_orig < thresh) * im_orig).astype(np.float)
    background_mean = ((im_orig < thresh) * mean).astype(np.float)
    new_img = foreground.astype(
        np.float) + background.astype(np.float) + background_mean.astype(np.float)
    return new_img


def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(
        image, rot_mat, image.shape[1::-1])
    return result


def translation(im, x, y):
    T = np.float32([[1, 0, x], [0, 1, y]])
    img_translation = cv2.warpAffine(im, T, (im.shape[1], im.shape[0]))
    return img_translation


def rotation(im_orig, rot):
    foreground = ((im_orig > thresh) * im_orig).astype(np.float)
    background = ((im_orig < thresh) * im_orig).astype(np.float)
    rotated_fg = rotateImage(foreground, rot)
    background = ((rotated_fg < thresh) * background).astype(np.float)
    new_img = rotated_fg + background
    return new_img


def radar_data_aug(im_orig, class_name, curr_range, new_range,
                   azi_res, stddev, init_bg_mean, rot, tx, ty):
    new_im = rotation(im_orig, rot)
    new_im = translation(new_im, tx, ty)
    new_im = attenuation_dataaug(
        new_im, class_name, curr_range, new_range)
    new_im = speckle_noise_dataaug(new_im, stddev)
    new_im = fill_zeros_with_gaussian_noise(new_im, bg_mean, bg_stddev)
    new_im = change_res(new_im, curr_range, new_range, azi_res)
    new_im = new_bg(new_im, init_bg_mean)

    return new_im


fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
new_im = radar_data_aug(
    im_orig, class_name, curr_range, new_range, azi_res, stddev, init_bg_mean, rot, tx, ty)
ax.imshow(new_im, vmin=0, vmax=255)
plt.axis('off')

axcolor = 'lightgoldenrodyellow'
axbg_noise = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)
axnoise = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)
axrot = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
axrange = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
axtx = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
axty = plt.axes([0.25, 0.0, 0.65, 0.03], facecolor=axcolor)

sbg_noise = Slider(axbg_noise, 'Background Noise',
                   -30.0, 30.0, valinit=0, valstep=0.1)
snoise = Slider(axnoise, 'Speckle Noise', 0.0, 0.2, valinit=0, valstep=0.01)
srot = Slider(axrot, 'Rotation', 0.0, 360.0, valinit=0, valstep=1)
srange = Slider(axrange, 'Range', 0.0, 15.0, valinit=0, valstep=0.1)
stx = Slider(axtx, 'Translation X', -100, 100, valinit=0, valstep=1)
sty = Slider(axty, 'Translation Y', -100, 100, valinit=0, valstep=0.1)


def update(val):
    init_bg_mean = sbg_noise.val
    stddev = snoise.val
    rot = srot.val
    new_range = srange.val
    tx = stx.val
    ty = sty.val

    new_im = radar_data_aug(
        im_orig, class_name, curr_range, new_range, azi_res, stddev, init_bg_mean, rot, tx, ty)
    ax.imshow(new_im, vmin=0, vmax=255)
    fig.canvas.draw_idle()


sbg_noise.on_changed(update)
snoise.on_changed(update)
srot.on_changed(update)
srange.on_changed(update)
stx.on_changed(update)
sty.on_changed(update)

plt.show()
