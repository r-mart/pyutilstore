import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches, patheffects
import cv2

# Visualization

def show_img_with_rois(img, rois, figsize=None, ax=None, color='white', save=False):
    
    if save:
        fig, ax = show_img(img, figsize=figsize, ax=ax, return_fig=save)
    else:
        ax = show_img(img, figsize=figsize, ax=ax, return_fig=save)
        
    for box in rois:
        box = bb_hw(box)
        draw_rect(ax, box, color=color)
        
    if save:
        fig.savefig('plot.png', bbox_inches='tight', pad_inches=0)


def show_img_with_seg(im, seg, figsize=None):
    fig, axes = plt.subplots(ncols=2, figsize=figsize)

    axes[0] = show_img(im, ax=axes[0])
    axes[1] = show_seg(seg, ax=axes[1])   


def show_img(im, figsize=None, ax=None, title=None, return_fig=False):
    dim = len(im.shape)
    assert (dim == 2 or dim == 3), "Image has to be represented by a 2D or 3D Numpy array"
    
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    if dim == 2:        
        ax.imshow(im, cmap='gray', vmin=0, vmax=255)
    else:
        ax.imshow(im)
    ax.axis('off')
    
    if title:
        ax.set_title(title, fontsize=18)

    if return_fig:
        return fig, ax        
    
    return ax  


def show_seg(im, figsize=None, ax=None, title=None, return_fig=False):
    
    dim = len(im.shape)
    assert (dim == 2 or dim == 3), "Image has to be represented by a 2D or 3D Numpy array"
    
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    if dim == 2:        
        ax.imshow(im, cmap='binary', vmin=np.min(im), vmax=np.max(im))
    else:
        ax.imshow(im)
    ax.get_xaxis().set_ticks([])        
    ax.get_yaxis().set_ticks([])        
    # ax.axis('off')
    
    if title:
        ax.set_title(title, fontsize=18)
        
    if return_fig:
        return fig, ax
    
    return ax      


def draw_rect(ax, b, color='white'):
    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor=color, lw=2))
    draw_outline(patch, 4)

    
def draw_text(ax, xy, txt, sz=14, color='white'):
    text = ax.text(*xy, txt,
        verticalalignment='top', color=color, fontsize=sz, weight='bold')
    draw_outline(text, 1)
    
    
def draw_outline(o, lw):
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])   
    
    
def bb_hw(a): 
    """ Convert bboxes from [y1, x1, y2, x2] to Matplotlib standard: [x1, y1, w, h] """

    return np.array([a[1],a[0],a[3]-a[1]+1,a[2]-a[0]+1])     


# Relational Databases

def blob_to_img(blob_data):  
    img_en = bytearray(blob_data)
    img_en = np.asarray(img_en, dtype=np.uint8)
    img = cv2.imdecode(img_en, cv2.IMREAD_GRAYSCALE)  
    
    return img  


# Image Processing

def get_translation_btw_images(tmpl, img):
    """ Determine translation between two shifted images of the same entity

    tmpl: numpy image of the reference entity
    img: numpy image of the shifted entity

    Tested with grayscale images only!
    Could be easily extended to get a full affine transforamtion
    """

    t_h, t_w = tmpl.shape[:2]
    img = cv2.resize(img, (t_w, t_h), interpolation = cv2.INTER_AREA)    

    good_match_thresh = 0.75
    feat_detector = cv2.AKAZE_create()
    matcher = cv2.BFMatcher()

    kps1, des1 = feat_detector.detectAndCompute(img, None)
    kps2, des2 = feat_detector.detectAndCompute(tmpl, None)

    matches = matcher.knnMatch(des1, des2, k=2)    

    # remove ambiguous matches  
    good_matches = []
    for m,n in matches:
        if m.distance < good_match_thresh * n.distance:
            good_matches.append([m])  

    if len(good_matches) > 0:
        points1 = np.float32([kps1[m[0].queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        points2 = np.float32([kps2[m[0].trainIdx].pt for m in good_matches]).reshape(-1,1,2)       

        mat, _ = cv2.estimateAffine2D(points1, points2)
    else:
        mat = None
 
    if mat is None:
        t_vec = [0, 0]
    else:
        t_vec = mat[:,2]
        t_vec = [int(round(t)) for t in t_vec]
    
    return t_vec   


def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    
    if len(im.shape) < 2 or len(im.shape) > 3:
        raise ValueError('The image to be cropped is expected to have dimension 2 or 3.')    

    h,w = im.shape[:2]

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)         


def crop_to_roi_with_margin(img, roi, margin):
    """ Crops an image to a given ROI while including margin 
    
    img: numpy image
    roi: numpy array of a bbox given by two points: [y1, x1, y2, x2]
    margin: (int) margin in pixel which will be added to the roi
    """
    
    local_area = roi.copy()
    local_area[:2] = local_area[:2] - margin
    local_area[2:] = local_area[2:] + margin    

    y1, x1, y2, x2 = local_area
    img_crop = img[y1:y2,x1:x2]   
    
    local_roi = roi - np.array([y1, x1, y1, x1])    
    
    return img_crop, local_roi 


def crop_image_to_points(img, points): 
    """ Crops image to a rectangular bbox determined by the hull of an array of points
    
    img: numpy image
    points: numpy array of points: [[p0_x, p0_y], [p1_x, p1_y], ..., [pn_x, pn_y]]
     """  
    
    x1 = np.min(points[:, 0])
    y1 = np.min(points[:, 1])

    x2 = np.max(points[:, 0])
    y2 = np.max(points[:, 1])    

    img_crop = img[y1:y2, x1:x2]

    return img_crop


# Geometry

def extract_rectangle(tetragons):
    """ Returns rectangles which embed the given tetragons

    Input: Array of 8 coordinates defining a tetragon: Either [p0_col, p0_row, ..., p3_col, p3_row] or [[p0_col, p0_row], ..., [p3_col, p3_row]]
    Output: List of 4 values defining a rectangle: (p0_col, p0_row, width, height)
    where p0 is the upper left corner
    """

    if tetragons is None:
        return None

    rectangles = np.zeros((len(tetragons), 4), np.int32)

    for i, tetragon in enumerate(tetragons):

        if len(tetragon.shape) > 1:
            tetragon = tetragon.flatten()
        
        col_pos = tetragon[range(0, len(tetragon), 2)]
        row_pos = tetragon[range(1, len(tetragon), 2)]

        min_col = np.maximum(0, round(np.min(col_pos)))
        min_row = np.maximum(0, round(np.min(row_pos)))
        max_col = np.maximum(0, round(np.max(col_pos)))
        max_row = np.maximum(0, round(np.max(row_pos)))

        rectangles[i] = min_col, min_row, max_col-min_col, max_row-min_row

    return rectangles


def get_normal_form(lines):
    """ Expects lines of the shape [x1, y1, x2, y2]. Returns the lines in normal form [rho, theta]. """

    lines_normal = np.zeros((len(lines), 2))

    for i, line in enumerate(lines):
        x1,y1,x2,y2 = line

        if y1 == y2: # horizontal
            theta = np.pi/2
            rho = y1

        else:
            theta = np.arctan( (x2-x1) / (y1-y2) )
            rho = x1 * np.cos(theta) + y1 * np.sin(theta)

            if theta < 0:
                theta = np.pi + theta
                rho = -rho

        lines_normal[i] = rho, theta

    return lines_normal    


def intersect(A,B,C,D):
    """ Test whether line segments AB and CD intersect    

    Returns true if they intersect, otherwise false

    If segments AB and CD intersect then only ACD or BCD can be counterclockwise, not both
    """
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


def ccw(A,B,C):
    """ Tests whether the three given points are in counterclockwise order

    Uses the fact that for points in counterclockwise order the slope of AB is less than the slope of the AC
    """
    A_x, A_y = A
    B_x, B_y = B
    C_x, C_y = C

    return (C_y-A_y) * (B_x-A_x) > (B_y-A_y) * (C_x-A_x)        


# PyTorch

import torch
from torchvision import transforms 

def get_features(image, model, target_layers):
    """ Get features of an image at different layers while traversing a model 

    image: pytorch image tensor
    model: pytorch model
    target_layers: dictionary of the shape {'l_idx': 'l_name', ...} 
    """

    features = {}
    x = image 
    for name, layer in model._modules.items():
        x = layer(x) # forward through model
        if name in target_layers:
            features[target_layers[name]] = x    

    return features


def gram_matrix(tensor):

    _, d, h, w = tensor.shape
    tensor = tensor.view(d, h * w)

    return torch.mm(tensor, tensor.t())    


def load_image(img_path, max_size=400, shape=None):
    """ Load an image from a path and convert it to an pytorch tensor 
    
    Intended to be used for natural images like ImageNet. Otherwise mean and std in Normalize should be adapted
    """
    
    image = Image.open(img_path).convert('RGB')
    
    # reduce size to speed up processing
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape
        
    in_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])

    # tensors also have an alpha channel (index 4), create dim for batch size
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    
    return image


def convert_image_np(img):
    """Convert a Tensor to numpy image 
    
    Intended to be used for natural images like ImageNet. Otherwise mean and std should be adapted
    """
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    img = img*255

    return img


# TensorFlow

import tensorflow as tf

def get_img_from_tensor(img_t):    
    img = img_t.numpy()    
    
    # drop channel for gray-scale images
    if len(img.shape) == 3 and img.shape[2] == 1:
        img = np.squeeze(img, axis=2)    
    
    img = img/2+0.5
    img = img*255
    return(img.astype(np.uint8))


def get_seg_from_tensor(seg_t):
    """ Get segmentation numpy image from the output of a segmentation network (e.g. U-Net) 
    
    seg_t: tf tensor of shape [H, W, N_classes] where the last dimension contains the probability 
        of the pixel at this position to belong to each of the classes
    """

    seg = seg_t.numpy()        
    seg = np.argmax(seg, axis=-1)    

    return(seg.astype(np.uint8))    


def get_resized_tensor_from_img(img, tensor_shape):
    img_t = tf.convert_to_tensor(img, dtype=tf.uint8)

    img_t = img_t / tf.constant(255, dtype=tf.uint8)

    if len(img_t.shape) == 2:
        img_t = tf.expand_dims(img_t, axis=-1)
        img_t = tf.tile(img_t, [1 , 1, 3])

    img_t = tf.image.resize(img_t, tensor_shape)    
    # transform image range from (0, 1) to (-1, 1), necessary for most pretrained Keras models
    img_t = 2*img_t-1  

    return img_t    