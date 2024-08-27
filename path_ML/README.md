```python
from pathlib import Path
from pathml.core import HESlide, SlideDataset
from histolab.slide import Slide
import os
```


```python
filepath = os.getcwd()
process_path = os.path.join(filepath, 'he_test')
filepath = "HE_Hamamatsu.ndpi"
init_slide = Slide(filepath, process_path)
```


```python
print(f"Slide name: {init_slide.name}")
print(f"Levels: {init_slide.levels}")
print(f"Dimensions at level 0: {init_slide.dimensions}")
print(f"Dimensions at level 1: {init_slide.level_dimensions(level=1)}")
print(f"Dimensions at level 2: {init_slide.level_dimensions(level=2)}")
print(f"Dimensions at level 3: {init_slide.level_dimensions(level=3)}")
print(f"Dimensions at level 4: {init_slide.level_dimensions(level=4)}")
print(f"Dimensions at level 5: {init_slide.level_dimensions(level=5)}")
```

    Slide name: OS-1
    Levels: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    Dimensions at level 0: (122880, 110592)
    Dimensions at level 1: (61440, 55296)
    Dimensions at level 2: (30720, 27648)
    Dimensions at level 3: (15360, 13824)
    Dimensions at level 4: (7680, 6912)
    Dimensions at level 5: (3840, 3456)



```python
init_slide.thumbnail
```




    
![png](output_3_0.png)
    




```python
from histolab.masks import TissueMask
import numpy as np
all_tissue_mask = TissueMask()
init_slide.locate_mask(all_tissue_mask)
```




    
![png](output_4_0.png)
    




```python
binary_mask: np.ndarray = all_tissue_mask(init_slide)
```


```python
from matplotlib import pyplot as plt
plt.imshow(binary_mask, interpolation='nearest')
plt.show()
```


    
![png](output_6_0.png)
    



```python
from histolab.util import regions_from_binary_mask
regions = regions_from_binary_mask(binary_mask)
```


```python
import numpy as np
import matplotlib.pyplot as plt

from pathml.core import HESlide
from pathml.preprocessing import StainNormalizationHE
```


```python
wsi = HESlide("HE_Hamamatsu.ndpi")
region = wsi.slide.extract_region(location = (10000, 10000), size = (1000, 1000))
plt.imshow(region)
```




    <matplotlib.image.AxesImage at 0x7fca048bb310>




    
![png](output_9_1.png)
    



```python
fig, axarr = plt.subplots(nrows=1, ncols=3, figsize=(10, 7.5))

method = "macenko"
fontsize = 12
for j, target in enumerate(["normalize", "hematoxylin", "eosin"]):
    normalizer = StainNormalizationHE(target = target, stain_estimation_method = "macenko")
    im = normalizer.F(region)
    ax = axarr[j]
    # plot results
    ax.imshow(im)
    ax.set_xlabel(f"{target}", fontsize=fontsize)
    if j == 0:
        ax.set_ylabel(f"{method} method", fontsize=fontsize)

plt.tight_layout()
plt.show()
```


    
![png](output_10_0.png)
    



```python
import numpy as np
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from torch.optim.lr_scheduler import StepLR
import albumentations as A
```


```python
from pathml.datasets.pannuke import PanNukeDataModule
from pathml.ml.hovernet import HoVerNet, loss_hovernet, post_process_batch_hovernet
from pathml.ml.utils import wrap_transform_multichannel, dice_score
from pathml.utils import plot_segmentation

```


```python
import matplotlib.pyplot as plt
import copy

from pathml.core import HESlide, Tile, types
from pathml.utils import plot_mask, RGB_to_GREY
from pathml.preprocessing import (
    BoxBlur, GaussianBlur, MedianBlur,
    NucleusDetectionHE, StainNormalizationHE, SuperpixelInterpolation,
    ForegroundDetection, TissueDetectionHE, BinaryThreshold,
    MorphClose, MorphOpen
)

fontsize = 14

```


```python
def smalltile():
    # convenience function to create a new tile
    return Tile(region, coords = (0, 0), name = "testregion", slide_type = types.HE)
```


```python
tile = smalltile()
nucleus_detection = NucleusDetectionHE(mask_name = "detect_nuclei")
nucleus_detection.apply(tile)

fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=(8, 8))
axarr[0].imshow(tile.image)
axarr[0].set_title("Original Image", fontsize=fontsize)
axarr[1].imshow(tile.masks["detect_nuclei"])
axarr[1].set_title("Nucleus Detection", fontsize=fontsize)
for ax in axarr.ravel():
    ax.set_yticks([])
    ax.set_xticks([])
plt.tight_layout()
plt.show()
```


    
![png](output_15_0.png)
    



```python
fig, ax = plt.subplots(figsize=(7, 7))
plot_mask(im = tile.image, mask_in=tile.masks["detect_nuclei"], ax = ax)
plt.title("Overlay", fontsize = fontsize)
plt.axis('off')
plt.show()
```


    
![png](output_16_0.png)
    



```python
n_classes_pannuke = 6

# data augmentation transform
hover_transform = A.Compose(
    [A.VerticalFlip(p=0.5),
     A.HorizontalFlip(p=0.5),
     A.RandomRotate90(p=0.5),
     A.GaussianBlur(p=0.5),
     A.MedianBlur(p=0.5, blur_limit=5)],
    additional_targets = {f"mask{i}" : "mask" for i in range(n_classes_pannuke)}
)

transform = wrap_transform_multichannel(hover_transform)

```


```python


import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import seaborn
import cv2
import torch

from PIL import Image
from tqdm import tqdm


```


```python


!git clone https://github.com/vqdang/hover_net.git
%cd hover_net


```

    Cloning into 'hover_net'...
    remote: Enumerating objects: 2032, done.[K
    remote: Counting objects: 100% (504/504), done.[K
    remote: Compressing objects: 100% (94/94), done.[K
    remote: Total 2032 (delta 431), reused 411 (delta 410), pack-reused 1528[K
    Receiving objects: 100% (2032/2032), 40.39 MiB | 20.14 MiB/s, done.
    Resolving deltas: 100% (1249/1249), done.
    /home/gazal/Documents/hover_net



```python
%%writefile dataset.py

import glob
import cv2
import numpy as np
import scipy.io as sio


class __AbstractDataset(object):
    """Abstract class for interface of subsequent classes.
    Main idea is to encapsulate how each dataset should parse
    their images and annotations.
    
    """

    def load_img(self, path):
        raise NotImplementedError

    def load_ann(self, path, with_type=False):
        raise NotImplementedError


####
class __PanNuke(__AbstractDataset):

    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_ann(self, path, with_type=False):
        # assumes that ann is HxW
        assert not with_type, "Not support"
        ann_inst = sio.loadmat(path)["inst_map"]
        ann_inst = ann_inst.astype("int32")
        ann = np.expand_dims(ann_inst, -1)
        return ann


####
def get_dataset(name):
    """Return a pre-defined dataset object associated with `name`."""
    name_dict = {
        "pannuke": lambda: __PanNuke(),
    }
    if name.lower() in name_dict:
        return name_dict[name]()
    else:
        assert False, "Unknown dataset `%s`" % name
```

    Overwriting dataset.py



```python
%%writefile dataset.py

import glob
import cv2
import numpy as np
import scipy.io as sio


class __AbstractDataset(object):
    """Abstract class for interface of subsequent classes.
    Main idea is to encapsulate how each dataset should parse
    their images and annotations.
    
    """

    def load_img(self, path):
        raise NotImplementedError

    def load_ann(self, path, with_type=False):
        raise NotImplementedError


####
class __PanNuke(__AbstractDataset):

    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_ann(self, path, with_type=False):
        # assumes that ann is HxW
        assert not with_type, "Not support"
        ann_inst = sio.loadmat(path)["inst_map"]
        ann_inst = ann_inst.astype("int32")
        ann = np.expand_dims(ann_inst, -1)
        return ann


####
def get_dataset(name):
    """Return a pre-defined dataset object associated with `name`."""
    name_dict = {
        "pannuke": lambda: __PanNuke(),
    }
    if name.lower() in name_dict:
        return name_dict[name]()
    else:
        assert False, "Unknown dataset `%s`" % name
```

    Overwriting dataset.py



```python
from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
from tiatoolbox.utils.misc import imread
from tiatoolbox.wsicore.wsireader import WSIReader

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, glob

mpl.rcParams['figure.dpi'] = 300 # for high resolution figure in notebook
plt.rcParams.update({'font.size': 5})
```


```python
from tiatoolbox.utils.misc import download_data

# These file name are used for the experimenets
img_file_name = "sample_tile.png"
wsi_file_name = "sample_wsi.svs"


print('Download has started. Please wait...')

# Downloading sample image tile
download_data("https://tiatoolbox.dcs.warwick.ac.uk/sample_imgs/breast_tissue_crop.png", img_file_name)

# Downloading sample whole-slide image
download_data("https://tiatoolbox.dcs.warwick.ac.uk/sample_wsis/wsi4_12k_12k.svs", wsi_file_name)

print('Download is complete.')
```

    Download has started. Please wait...
    Download is complete.



```python
# Tile prediction
inst_segmentor = NucleusInstanceSegmentor(
    pretrained_model="hovernet_fast-pannuke",
    num_loader_workers=2,
    num_postproc_workers=2,
    batch_size=4,
)

tile_output = inst_segmentor.predict(
        [img_file_name],
        save_dir="he_test_new_/",
        mode="tile",
        on_gpu=False,
        crash_on_exception=True,
    )
```

    |2024-03-20|15:08:08.311| [WARNING] WSIPatchDataset only reads image tile at `units="baseline"`. Resolutions will be converted to baseline value.
    |2024-03-20|15:08:08.336| [WARNING] WSIPatchDataset only reads image tile at `units="baseline"`. Resolutions will be converted to baseline value.
    |2024-03-20|15:08:08.374| [WARNING] Raw data is None.
    |2024-03-20|15:08:08.374| [WARNING] Unknown scale (no objective_power or mpp)



    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    Cell In[18], line 9
          1 # Tile prediction
          2 inst_segmentor = NucleusInstanceSegmentor(
          3     pretrained_model="hovernet_fast-pannuke",
          4     num_loader_workers=2,
          5     num_postproc_workers=2,
          6     batch_size=4,
          7 )
    ----> 9 tile_output = inst_segmentor.predict(
         10         [img_file_name],
         11         save_dir="he_test_new_/",
         12         mode="tile",
         13         on_gpu=False,
         14         crash_on_exception=True,
         15     )


    File ~/.local/lib/python3.8/site-packages/tiatoolbox/models/engine/semantic_segmentor.py:1399, in SemanticSegmentor.predict(self, imgs, masks, mode, ioconfig, patch_input_shape, patch_output_shape, stride_shape, resolution, units, save_dir, on_gpu, crash_on_exception)
       1396 # ? what will happen if this crash midway?
       1397 # => may not be able to retrieve the result dict
       1398 for wsi_idx, img_path in enumerate(imgs):
    -> 1399     self._predict_wsi_handle_exception(
       1400         imgs=imgs,
       1401         wsi_idx=wsi_idx,
       1402         img_path=img_path,
       1403         mode=mode,
       1404         ioconfig=ioconfig,
       1405         save_dir=save_dir,
       1406         crash_on_exception=crash_on_exception,
       1407     )
       1409 # clean up the cache directories
       1410 try:


    File ~/.local/lib/python3.8/site-packages/tiatoolbox/models/engine/semantic_segmentor.py:1249, in SemanticSegmentor._predict_wsi_handle_exception(self, imgs, wsi_idx, img_path, mode, ioconfig, save_dir, crash_on_exception)
       1247 wsi_save_path = save_dir.joinpath(f"{wsi_idx}")
       1248 if crash_on_exception:
    -> 1249     raise err  # noqa: TRY201
       1250 logging.exception("Crashed on %s", wsi_save_path)


    File ~/.local/lib/python3.8/site-packages/tiatoolbox/models/engine/semantic_segmentor.py:1225, in SemanticSegmentor._predict_wsi_handle_exception(self, imgs, wsi_idx, img_path, mode, ioconfig, save_dir, crash_on_exception)
       1223 try:
       1224     wsi_save_path = save_dir / f"{wsi_idx}"
    -> 1225     self._predict_one_wsi(wsi_idx, ioconfig, str(wsi_save_path), mode)
       1227     # Do not use dict with file name as key, because it can be
       1228     # overwritten. It may be user intention to provide files with a
       1229     # same name multiple times (maybe they have different root path)
       1230     self._outputs.append([str(img_path), str(wsi_save_path)])


    File ~/.local/lib/python3.8/site-packages/tiatoolbox/models/engine/nucleus_instance_segmentor.py:741, in NucleusInstanceSegmentor._predict_one_wsi(self, wsi_idx, ioconfig, save_path, mode)
        739 tile_patch_inputs = patch_inputs[sel_indices]
        740 tile_patch_outputs = patch_outputs[sel_indices]
    --> 741 self._to_shared_space(wsi_idx, tile_patch_inputs, tile_patch_outputs)
        743 tile_infer_output = self._infer_once()
        745 self._process_tile_predictions(
        746     ioconfig,
        747     tile_bounds,
       (...)
        750     tile_infer_output,
        751 )


    File ~/.local/lib/python3.8/site-packages/tiatoolbox/models/engine/nucleus_instance_segmentor.py:617, in NucleusInstanceSegmentor._to_shared_space(self, wsi_idx, patch_inputs, patch_outputs)
        615 patch_inputs = torch.from_numpy(patch_inputs).share_memory_()
        616 patch_outputs = torch.from_numpy(patch_outputs).share_memory_()
    --> 617 self._mp_shared_space.patch_inputs = patch_inputs
        618 self._mp_shared_space.patch_outputs = patch_outputs
        619 self._mp_shared_space.wsi_idx = torch.Tensor([wsi_idx]).share_memory_()


    File /usr/lib/python3.8/multiprocessing/managers.py:1143, in NamespaceProxy.__setattr__(self, key, value)
       1141     return object.__setattr__(self, key, value)
       1142 callmethod = object.__getattribute__(self, '_callmethod')
    -> 1143 return callmethod('__setattr__', (key, value))


    File /usr/lib/python3.8/multiprocessing/managers.py:834, in BaseProxy._callmethod(self, methodname, args, kwds)
        831     self._connect()
        832     conn = self._tls.connection
    --> 834 conn.send((self._id, methodname, args, kwds))
        835 kind, result = conn.recv()
        837 if kind == '#RETURN':


    File /usr/lib/python3.8/multiprocessing/connection.py:206, in _ConnectionBase.send(self, obj)
        204 self._check_closed()
        205 self._check_writable()
    --> 206 self._send_bytes(_ForkingPickler.dumps(obj))


    File /usr/lib/python3.8/multiprocessing/reduction.py:51, in ForkingPickler.dumps(cls, obj, protocol)
         48 @classmethod
         49 def dumps(cls, obj, protocol=None):
         50     buf = io.BytesIO()
    ---> 51     cls(buf, protocol).dump(obj)
         52     return buf.getbuffer()


    File ~/.local/lib/python3.8/site-packages/torch/multiprocessing/reductions.py:295, in reduce_tensor(tensor)
        294 def rebuild_storage_filename(cls, manager, handle, size):
    --> 295     storage = storage_from_cache(cls, handle)
        296     if storage is not None:
        297         return storage._shared_decref()


    ModuleNotFoundError: No module named 'torch.nested._internal'



```python
from histolab.data import prostate_tissue, ovarian_tissue
```


```python
prostate_svs, prostate_path = prostate_tissue()
ovarian_svs, ovarian_path = ovarian_tissue()
```

    Downloading file 'tcga/prostate/TCGA-CH-5753-01A-01-BS1.4311c533-f9c1-4c6f-8b10-922daa3c2e3e.svs' from 'https://api.gdc.cancer.gov/data/5a8ce04a-0178-49e2-904c-30e21fb4e41e' to '/home/gazal/.cache/histolab-images/0.6.0'.
    Downloading file 'tcga/ovarian/TCGA-13-1404-01A-01-TS1.cecf7044-1d29-4d14-b137-821f8d48881e.svs' from 'https://api.gdc.cancer.gov/data/e968375e-ef58-4607-b457-e6818b2e8431' to '/home/gazal/.cache/histolab-images/0.6.0'.



```python
from histolab.slide import Slide
```


```python
import os

BASE_PATH = os.getcwd()

PROCESS_PATH_PROSTATE = os.path.join(BASE_PATH, 'prostate', 'processed')
PROCESS_PATH_OVARIAN = os.path.join(BASE_PATH, 'ovarian', 'processed')

prostate_slide = Slide(prostate_path, processed_path=PROCESS_PATH_PROSTATE)
ovarian_slide = Slide(ovarian_path, processed_path=PROCESS_PATH_OVARIAN)
```


```python
print(f"Slide name: {prostate_slide.name}")
print(f"Levels: {prostate_slide.levels}")
print(f"Dimensions at level 0: {prostate_slide.dimensions}")
print(f"Dimensions at level 1: {prostate_slide.level_dimensions(level=1)}")
print(f"Dimensions at level 2: {prostate_slide.level_dimensions(level=2)}")
```

    Slide name: TCGA-CH-5753-01A-01-BS1.4311c533-f9c1-4c6f-8b10-922daa3c2e3e
    Levels: [0, 1, 2]
    Dimensions at level 0: (16000, 15316)
    Dimensions at level 1: (4000, 3829)
    Dimensions at level 2: (2000, 1914)



```python


prostate_slide.thumbnail
prostate_slide.show()


```


```python
from histolab.tiler import GridTiler
```


```python
grid_tiles_extractor = GridTiler(
   tile_size=(512, 512),
   level=0,
   check_tissue=True, # default
   pixel_overlap=0, # default
   prefix="grid/", # save tiles in the "grid" subdirectory of slide's processed_path
   suffix=".png" # default
)
```


```python
grid_tiles_extractor.locate_tiles(
    slide=ovarian_slide,
    scale_factor=64,
    alpha=64,
    outline="#046C4C",
)
```




    
![png](output_33_0.png)
    




```python
grid_tiles_extractor.extract(init_slide)
```


```python

```
