from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib.pyplot as plt


from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist.models import Config2D, StarDist2D, StarDistData2D

import argparse

#Create the parser
my_parser = argparse.ArgumentParser(description='Train the stardist segmentation algorithm on 2D images')

#Add the arguments
my_parser.add_argument('--path',type=str,help='Training folder path')
my_parser.add_argument('--n_rays',type=int,default=32,help='Number of rays for the convex poligons')
my_parser.add_argument('--use_gpu',type=bool,default=True,help='Use GPU or not')
my_parser.add_argument('--grid',type=list,default=(2,2),help='Subsampled grid for increased efficiency and larger field of view')
my_parser.add_argument('--limit_gpu_mem',type=float,default=0.8,help='Limit for the GPU memory usage')
my_parser.add_argument('--model_dir',type=str,default='models',help='Models directory')
my_parser.add_argument('--model_name',type=str,default='stardist',help='Name of the model')
my_parser.add_argument('--quick_demo',type=bool,default='False',help='Run quick training demo (True) or full training (False)')
my_parser.add_argument('--epochs',type=int,default=400,help='Number of training epochs')
my_parser.add_argument('--steps_per_epoch',type=int,default=100,help='Steps per training epoch')


args = my_parser.parse_args()


trainpath=args.path
np.random.seed(42)
lbl_cmap = random_label_cmap()


X = sorted(glob(trainpath+'/images/*.tif'))
Y = sorted(glob(trainpath+'/masks/*.tif'))
assert all(Path(x).name==Path(y).name for x,y in zip(X,Y))

X = list(map(imread,X))
Y = list(map(imread,Y))
n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]



axis_norm = (0,1)   # normalize channels independently
# axis_norm = (0,1,2) # normalize channels jointly
if n_channel > 1:
    print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
    sys.stdout.flush()

X = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(X)]
Y = [fill_label_holes(y) for y in tqdm(Y)]


assert len(X) > 1, "not enough training data"
rng = np.random.RandomState(42)
ind = rng.permutation(len(X))
n_val = max(1, int(round(0.15 * len(ind))))
ind_train, ind_val = ind[:-n_val], ind[-n_val:]
X_val, Y_val = [X[i] for i in ind_val]  , [Y[i] for i in ind_val]
X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train] 
print('number of images: %3d' % len(X))
print('- training:       %3d' % len(X_trn))
print('- validation:     %3d' % len(X_val))


#i = min(9, len(X)-1)
#img, lbl = X[i], Y[i]
#assert img.ndim in (2,3)
#img = img if (img.ndim==2 or img.shape[-1]==3) else img[...,0]
#plt.figure(figsize=(16,10))
#plt.subplot(121); plt.imshow(img,cmap='gray');   plt.axis('off'); plt.title('Raw image')
#plt.subplot(122); plt.imshow(lbl,cmap=lbl_cmap); plt.axis('off'); plt.title('GT labels')
#None;



# 32 is a good default choice (see 1_data.ipynb)
n_rays = args.n_rays

# Use OpenCL-based computations for data generator during training (requires 'gputools')
use_gpu = args.use_gpu and gputools_available()

# Predict on subsampled grid for increased efficiency and larger field of view
grid = args.grid

conf = Config2D (
    n_rays       = n_rays,
    grid         = grid,
    use_gpu      = use_gpu,
    n_channel_in = n_channel,
)
print(conf)
vars(conf)


if use_gpu:
    from csbdeep.utils.tf import limit_gpu_memory
    # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
    limit_gpu_memory(args.limit_gpu_mem)




model = StarDist2D(conf, name=args.model_name, basedir=args.model_dir)


median_size = calculate_extents(list(Y), np.median)
fov = np.array(model._axes_tile_overlap('YX'))
if any(median_size > fov):
    print("WARNING: median object size larger than field of view of the neural network.")



augmenter = None

# def augmenter(x, y):
#     """Augmentation of a single input/label image pair.
#     x is an input image
#     y is the corresponding ground-truth label image
#     """
#     # modify a copy of x and/or y...
#     return x, y


quick_demo = args.quick_demo

if quick_demo:
    print (
        "NOTE: This is only for a quick demonstration!\n"
        "      Please set the variable 'quick_demo = False' for proper (long) training.",
        file=sys.stderr, flush=True
    )
    model.train(X_trn, Y_trn, validation_data=(X_val,Y_val), augmenter=augmenter,
                epochs=2, steps_per_epoch=10)

    print("====> Stopping training and loading previously trained demo model from disk.", file=sys.stderr, flush=True)
    model = StarDist2D(None, name='2D_demo', basedir='../../models/examples')
    model.basedir = None # to prevent files of the demo model to be overwritten (not needed for your model)
else:
    model.train(X_trn, Y_trn, validation_data=(X_val,Y_val), augmenter=augmenter,epochs=args.epochs, steps_per_epoch=args.steps_per_epoch)
None;


model.optimize_thresholds(X_val, Y_val)